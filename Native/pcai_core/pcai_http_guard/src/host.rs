//! `Host` header parsing and the allow-list that defeats DNS rebinding.
//!
//! In a rebinding attack the browser still believes it talks to `evil.example`, so every request
//! carries `Host: evil.example` even though the name now resolves to `127.0.0.1`. A server that
//! only accepts the names it is actually reached by cannot be driven that way.

use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use crate::SecurityError;

/// A host reduced to what is compared: a canonical IP, or a lowercase DNS name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum HostKey {
    Ip(IpAddr),
    Name(String),
}

/// Parse the authority of a `Host` header (or of a request URI) into a [`HostKey`], ignoring
/// the port.
///
/// Accepts `name`, `name:port`, `1.2.3.4`, `1.2.3.4:port`, `[v6]`, `[v6]:port` and, leniently,
/// an unbracketed IPv6 literal such as `::1`. Returns `None` for anything malformed: user info,
/// zone identifiers, empty or out-of-range ports, whitespace, odd characters.
pub(crate) fn parse_authority_host(value: &str) -> Option<HostKey> {
    if let Some(rest) = value.strip_prefix('[') {
        let (inner, after) = rest.split_once(']')?;
        if !after.is_empty() && !valid_port(after.strip_prefix(':')?) {
            return None;
        }
        let ip: Ipv6Addr = inner.parse().ok()?;
        return Some(HostKey::Ip(IpAddr::V6(ip).to_canonical()));
    }

    let host = match value.matches(':').count() {
        0 => value,
        1 => {
            let (host, port) = value.split_once(':')?;
            if !valid_port(port) {
                return None;
            }
            host
        }
        _ => {
            let ip: Ipv6Addr = value.parse().ok()?;
            return Some(HostKey::Ip(IpAddr::V6(ip).to_canonical()));
        }
    };

    if let Ok(ip) = host.parse::<Ipv4Addr>() {
        return Some(HostKey::Ip(IpAddr::V4(ip)));
    }
    let name = host.strip_suffix('.').unwrap_or(host);
    let plausible = !name.is_empty()
        && name.len() <= 253
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_'));
    plausible.then(|| HostKey::Name(name.to_ascii_lowercase()))
}

fn valid_port(port: &str) -> bool {
    !port.is_empty() && port.len() <= 5 && port.bytes().all(|byte| byte.is_ascii_digit()) && port.parse::<u16>().is_ok()
}

/// The set of hosts a server answers to.
#[derive(Debug, Clone)]
pub(crate) struct HostAllowList {
    allowed: HashSet<HostKey>,
}

impl HostAllowList {
    /// Loopback names, the bind address (unless it is a wildcard) and the configured extras.
    pub(crate) fn new(bind: IpAddr, extra: &[String]) -> Result<Self, SecurityError> {
        let mut allowed = HashSet::from([
            HostKey::Name("localhost".to_string()),
            HostKey::Ip(IpAddr::V4(Ipv4Addr::LOCALHOST)),
            HostKey::Ip(IpAddr::V6(Ipv6Addr::LOCALHOST)),
        ]);

        // `0.0.0.0` and `::` are not names any client should use, and browsers have treated
        // `0.0.0.0` as loopback, so a wildcard bind never adds itself.
        let bind = bind.to_canonical();
        if !bind.is_unspecified() {
            allowed.insert(HostKey::Ip(bind));
        }

        for entry in extra {
            let invalid = || SecurityError::InvalidAllowedHost { host: entry.clone() };
            let key = parse_authority_host(entry.trim()).ok_or_else(invalid)?;
            if matches!(&key, HostKey::Ip(ip) if ip.is_unspecified()) {
                return Err(invalid());
            }
            allowed.insert(key);
        }
        Ok(Self { allowed })
    }

    pub(crate) fn allows(&self, key: &HostKey) -> bool {
        self.allowed.contains(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(text: &str) -> HostKey {
        HostKey::Ip(text.parse::<IpAddr>().expect("literal").to_canonical())
    }

    fn name(text: &str) -> HostKey {
        HostKey::Name(text.to_string())
    }

    // Protects: the Host spellings real clients send, including the HVSocket proxy's forwarded
    // port. Detects: a parser that mishandles the port, brackets, case or the trailing dot.
    #[test]
    fn parses_hosts_with_and_without_ports() {
        let cases = [
            ("127.0.0.1", ip("127.0.0.1")),
            ("127.0.0.1:18080", ip("127.0.0.1")),
            ("localhost", name("localhost")),
            ("LocalHost:1", name("localhost")),
            ("localhost.:80", name("localhost")),
            ("[::1]", ip("::1")),
            ("[::1]:9", ip("::1")),
            ("::1", ip("::1")),
            ("[0:0:0:0:0:0:0:1]:80", ip("::1")),
            ("[::ffff:127.0.0.1]:80", ip("127.0.0.1")),
            ("Inference.LAN:8080", name("inference.lan")),
        ];
        for (input, expected) in cases {
            assert_eq!(parse_authority_host(input), Some(expected), "input {input:?}");
        }
    }

    // Protects: refusal of malformed authorities. Detects: lenient parsing that would let a
    // crafted value be read as loopback (user info, zones, junk after the bracket, bad ports).
    #[test]
    fn rejects_malformed_authorities() {
        for input in [
            "",
            ":80",
            "127.0.0.1:",
            "127.0.0.1:99999",
            "127.0.0.1:80x",
            "user@127.0.0.1",
            "[::1]x",
            "[::1",
            "[fe80::1%3]",
            "has space",
            "a/b",
            "a:b:c",
            "[127.0.0.1]",
        ] {
            assert_eq!(parse_authority_host(input), None, "input {input:?}");
        }
    }

    // Protects: look-alikes stay names, not IPs. Detects: lenient numeric parsing that would turn
    // `0x7f.0.0.1` or `2130706433` into 127.0.0.1.
    #[test]
    fn numeric_look_alikes_are_plain_names() {
        for input in ["0x7f.0.0.1", "2130706433", "127.1", "0177.0.0.1"] {
            assert!(
                matches!(parse_authority_host(input), Some(HostKey::Name(_))),
                "input {input:?}"
            );
        }
    }

    // Protects: the default allow-list. Detects: a missing loopback name, the wildcard bind IP
    // being added, or the real bind IP being left out.
    #[test]
    fn default_allow_list_contents() {
        let list = HostAllowList::new("127.0.0.1".parse().expect("literal"), &[]).expect("loopback list");
        for allowed in [name("localhost"), ip("127.0.0.1"), ip("::1")] {
            assert!(list.allows(&allowed), "{allowed:?}");
        }
        assert!(!list.allows(&name("evil.example")));

        let list = HostAllowList::new("192.168.1.20".parse().expect("literal"), &[]).expect("lan list");
        assert!(list.allows(&ip("192.168.1.20")));

        for wildcard in ["0.0.0.0", "::"] {
            let list = HostAllowList::new(wildcard.parse().expect("literal"), &[]).expect("wildcard list");
            assert!(
                !list.allows(&ip(wildcard)),
                "wildcard {wildcard} must not be allowed as a Host"
            );
        }
    }

    // Protects: validation of `allowed_hosts`. Detects: accepting entries that can never match
    // (wildcards, junk) or that re-open the `0.0.0.0` rebinding hole.
    #[test]
    fn configured_hosts_are_validated_and_normalised() {
        let loopback: IpAddr = "127.0.0.1".parse().expect("literal");
        let list = HostAllowList::new(loopback, &["Inference.LAN:8080".to_string(), "10.0.0.5".to_string()])
            .expect("valid extras");
        assert!(list.allows(&name("inference.lan")));
        assert!(list.allows(&ip("10.0.0.5")));

        for bad in ["*", "", "has space", "a/b", "0.0.0.0", "::", "[::]:80", "user@host"] {
            let outcome = HostAllowList::new(loopback, &[bad.to_string()]);
            assert!(
                matches!(outcome, Err(SecurityError::InvalidAllowedHost { .. })),
                "{bad:?} must be rejected"
            );
        }
    }
}
