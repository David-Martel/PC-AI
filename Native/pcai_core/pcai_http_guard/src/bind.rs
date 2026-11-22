//! Bind-address parsing and listener creation.
//!
//! Addresses used to be built with `format!("{}:{}", host, port)`, which cannot express an IPv6
//! literal (`::1:8080` is ambiguous) or a scoped link-local address (`fe80::1%12`), and which
//! handed any non-IP host to the system resolver. [`parse_bind_addr`] parses instead, and
//! [`bind_listener`] fails loudly with the address in the message.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV6};

/// Failure to turn configuration into a listening socket.
#[derive(Debug, thiserror::Error)]
pub enum BindError {
    /// The configured host was empty.
    #[error("the bind host is empty; use an IP address such as 127.0.0.1 or ::1")]
    Empty,

    /// The host is not an IP literal.
    #[error(
        "bind host {host:?} is not an IP literal; use an IPv4 address, an IPv6 address (with or without brackets, \
         optionally scoped like fe80::1%12) or `localhost`"
    )]
    NotAnIp {
        /// The host string as configured.
        host: String,
    },

    /// The `%scope` of an IPv6 literal is neither a number nor a known interface name.
    #[error("bind host {host:?} has an invalid IPv6 scope: {reason}")]
    InvalidScope {
        /// The host string as configured.
        host: String,
        /// What is wrong with the scope.
        reason: String,
    },

    /// The socket could not be bound (port in use, address not assigned, permission denied).
    #[error("failed to bind {addr}: {source}")]
    Listen {
        /// The address that was requested.
        addr: SocketAddr,
        /// The operating system error.
        #[source]
        source: std::io::Error,
    },
}

/// Parse a configured host and port into a [`SocketAddr`] without any name resolution.
///
/// Accepted hosts:
///
/// - an IPv4 literal (`127.0.0.1`, `0.0.0.0`);
/// - an IPv6 literal with or without brackets (`::1`, `[::1]`);
/// - a scoped IPv6 literal, where the scope is an interface index or an interface name
///   (`fe80::1%12`, `[fe80::1%eth0]`); the scope becomes the socket's `scope_id`;
/// - `localhost` (any case), which means `127.0.0.1`.
///
/// # Errors
///
/// Returns [`BindError`] for an empty host, a host that is not an IP literal (a DNS name would
/// make the bind address depend on the resolver), or an unknown IPv6 scope.
///
/// # Examples
///
/// ```
/// use pcai_http_guard::parse_bind_addr;
///
/// let v6 = parse_bind_addr("[::1]", 8080)?;
/// assert_eq!(v6.to_string(), "[::1]:8080");
///
/// let scoped = parse_bind_addr("fe80::1%3", 8080)?;
/// assert_eq!(scoped.to_string(), "[fe80::1%3]:8080");
///
/// assert!(parse_bind_addr("example.com", 8080).is_err());
/// # Ok::<(), pcai_http_guard::BindError>(())
/// ```
pub fn parse_bind_addr(host: &str, port: u16) -> Result<SocketAddr, BindError> {
    let trimmed = host.trim();
    if trimmed.is_empty() {
        return Err(BindError::Empty);
    }
    if trimmed.eq_ignore_ascii_case("localhost") {
        return Ok(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port));
    }

    let not_an_ip = || BindError::NotAnIp { host: host.to_string() };
    let inner = trimmed
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .unwrap_or(trimmed);

    if let Some((address, zone)) = inner.split_once('%') {
        let ip: Ipv6Addr = address.parse().map_err(|_| not_an_ip())?;
        let scope_id = resolve_scope(zone).map_err(|reason| BindError::InvalidScope {
            host: host.to_string(),
            reason,
        })?;
        return Ok(SocketAddr::V6(SocketAddrV6::new(ip, port, 0, scope_id)));
    }

    let ip: IpAddr = inner.parse().map_err(|_| not_an_ip())?;
    Ok(SocketAddr::new(ip, port))
}

/// Bind a TCP listener, naming the address in the error.
///
/// There is deliberately no fallback to another port or address: a server that silently listens
/// somewhere else is unreachable for its clients and hides the conflict.
///
/// # Errors
///
/// Returns [`BindError::Listen`] when the operating system refuses the bind.
pub async fn bind_listener(addr: SocketAddr) -> Result<tokio::net::TcpListener, BindError> {
    tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|source| BindError::Listen { addr, source })
}

/// Whether an address is a loopback address, treating IPv4-mapped IPv6 (`::ffff:127.0.0.1`) as
/// the IPv4 address it wraps.
///
/// # Examples
///
/// ```
/// use pcai_http_guard::is_loopback;
///
/// assert!(is_loopback("127.0.0.9".parse()?));
/// assert!(is_loopback("::ffff:127.0.0.1".parse()?));
/// assert!(!is_loopback("0.0.0.0".parse()?));
/// # Ok::<(), std::net::AddrParseError>(())
/// ```
#[must_use]
pub fn is_loopback(ip: IpAddr) -> bool {
    ip.to_canonical().is_loopback()
}

/// Resolve the text after `%` to an interface index.
fn resolve_scope(zone: &str) -> Result<u32, String> {
    if zone.is_empty() {
        return Err("the scope after `%` is empty".to_string());
    }
    if zone.bytes().all(|byte| byte.is_ascii_digit()) {
        return match zone.parse::<u32>() {
            Ok(0) => Err("scope id 0 means no scope; use a nonzero interface index".to_string()),
            Ok(index) => Ok(index),
            Err(_) => Err(format!("scope id {zone} is out of range")),
        };
    }
    interface_index(zone).ok_or_else(|| format!("no network interface is named {zone:?}"))
}

#[cfg(unix)]
fn interface_index(name: &str) -> Option<u32> {
    let name = std::ffi::CString::new(name).ok()?;
    // SAFETY: `name` is a valid NUL-terminated string that lives for the whole call, and
    // `if_nametoindex` only reads it.
    let index = unsafe { libc::if_nametoindex(name.as_ptr()) };
    (index != 0).then_some(index)
}

#[cfg(windows)]
fn interface_index(name: &str) -> Option<u32> {
    let name = std::ffi::CString::new(name).ok()?;
    // SAFETY: `name` is a valid NUL-terminated string that lives for the whole call, and
    // `if_nametoindex` only reads it.
    let index = unsafe { windows_sys::Win32::NetworkManagement::IpHelper::if_nametoindex(name.as_ptr().cast()) };
    (index != 0).then_some(index)
}

#[cfg(not(any(unix, windows)))]
fn interface_index(_name: &str) -> Option<u32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // Protects: every spelling a config file or the PowerShell generators can produce. Detects:
    // going back to `format!("{host}:{port}")`, which turns `::1` into the unparsable `::1:8080`
    // and cannot carry a scope. Needs: nothing but the parser.
    #[test]
    fn accepts_ip_literals_in_every_spelling() {
        let cases: [(&str, &str); 9] = [
            ("127.0.0.1", "127.0.0.1:80"),
            ("0.0.0.0", "0.0.0.0:80"),
            ("::1", "[::1]:80"),
            ("[::1]", "[::1]:80"),
            ("  [::1]  ", "[::1]:80"),
            ("2001:db8::7", "[2001:db8::7]:80"),
            ("localhost", "127.0.0.1:80"),
            ("LocalHost", "127.0.0.1:80"),
            ("::", "[::]:80"),
        ];
        for (host, expected) in cases {
            let parsed = parse_bind_addr(host, 80).unwrap_or_else(|error| panic!("{host:?}: {error}"));
            assert_eq!(parsed.to_string(), expected, "host {host:?}");
        }
    }

    // Protects: link-local binds such as the USB4 peer link. Detects: dropping the scope, which
    // makes Windows bind fail or pick the wrong interface.
    #[test]
    fn numeric_scope_becomes_scope_id() {
        for host in ["fe80::1%3", "[fe80::1%3]"] {
            let parsed = parse_bind_addr(host, 9).expect("scoped literal must parse");
            let SocketAddr::V6(v6) = parsed else {
                panic!("{host} must parse to an IPv6 address");
            };
            assert_eq!(v6.scope_id(), 3);
            assert_eq!(v6.port(), 9);
            assert_eq!(*v6.ip(), "fe80::1".parse::<Ipv6Addr>().expect("literal"));
        }
    }

    // Protects: loud failure for input that cannot be bound as written. Detects: a parser that
    // falls back to the resolver, accepts a port inside the host, or ignores a bad scope.
    #[test]
    fn rejects_invalid_hosts_and_scopes() {
        for host in [
            "",
            "   ",
            "example.com",
            "not a host",
            "127.0.0.1:8080",
            "[::1",
            "::1]",
            "1.2.3",
            "fe80::1%",
            "fe80::1%0",
            "fe80::1%99999999999",
            "fe80::1%definitely-not-an-interface",
            "127.0.0.1%3",
            "*",
        ] {
            let outcome = parse_bind_addr(host, 80);
            assert!(outcome.is_err(), "host {host:?} must be rejected, got {outcome:?}");
        }
    }

    // Protects: the error text, which is the only thing an operator sees at startup. Detects: an
    // error that omits the offending host.
    #[test]
    fn errors_name_the_offending_host() {
        let message = parse_bind_addr("not a host", 80).expect_err("must fail").to_string();
        assert!(message.contains("not a host"), "{message}");
        let message = parse_bind_addr("fe80::1%nope0", 80).expect_err("must fail").to_string();
        assert!(
            message.contains("fe80::1%nope0") && message.contains("scope"),
            "{message}"
        );
    }

    // Protects: loopback detection used by the "no token on a network bind" rule. Detects: a
    // check that misses 127.0.0.0/8 or IPv4-mapped IPv6, or that treats wildcards as loopback.
    #[test]
    fn loopback_classification() {
        for ip in ["127.0.0.1", "127.9.9.9", "::1", "::ffff:127.0.0.1"] {
            assert!(is_loopback(ip.parse().expect("literal")), "{ip}");
        }
        for ip in [
            "0.0.0.0",
            "::",
            "192.168.0.2",
            "fe80::1",
            "2001:db8::1",
            "::ffff:10.0.0.1",
        ] {
            assert!(!is_loopback(ip.parse().expect("literal")), "{ip}");
        }
    }

    // Protects: loud bind failures (FunctionGemma got the same fix in #408). Detects: a fallback
    // to another port, or an error that does not name the requested address. Needs: a loopback
    // socket held by this test.
    #[tokio::test]
    async fn occupied_address_fails_and_names_the_address() {
        let occupied = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind loopback");
        let addr = occupied.local_addr().expect("local addr");
        let error = bind_listener(addr)
            .await
            .expect_err("bind must fail while the port is held");
        assert!(error.to_string().contains(&addr.to_string()), "{error}");
        assert!(matches!(
            error,
            BindError::Listen { ref source, .. } if source.kind() == std::io::ErrorKind::AddrInUse
        ));
        drop(occupied);
    }

    // Protects: IPv6 loopback listening. Detects: a bind path that only handles IPv4. Needs: an
    // IPv6 loopback; hosts without one skip the assertion.
    #[tokio::test]
    async fn binds_ipv6_loopback_when_available() {
        let addr = parse_bind_addr("[::1]", 0).expect("literal");
        match bind_listener(addr).await {
            Ok(listener) => assert_eq!(listener.local_addr().expect("local addr").ip(), addr.ip()),
            Err(BindError::Listen { source, .. }) if source.kind() == std::io::ErrorKind::AddrNotAvailable => {}
            Err(other) => panic!("unexpected bind failure: {other}"),
        }
    }
}
