#!/bin/bash
echo "=== Test 1: Preflight checks ==="
CURL="$(command -v curl 2>/dev/null || echo /sbin/curl)"
echo "curl: ${CURL}"
test -x "${CURL}" && echo "PASS: curl executable" || echo "FAIL: curl not executable"
command -v openssl >/dev/null 2>&1 && echo "PASS: openssl" || echo "FAIL: openssl"
test -d /share/CACHEDEV1_DATA && echo "PASS: CACHEDEV1_DATA exists" || echo "FAIL: CACHEDEV1_DATA"
"${CURL}" -sSf -o /dev/null --connect-timeout 10 "https://acme-v02.api.letsencrypt.org/directory" 2>/dev/null && echo "PASS: LE reachable" || echo "FAIL: LE unreachable"

echo ""
echo "=== Test 2: JQ parsing ==="
JQ="/usr/local/sbin/jq"
echo '{"result":{"name":"dtmventures.com"}}' | "${JQ}" -r '.result.name' 2>/dev/null && echo "PASS: jq works" || echo "FAIL: jq"
echo '{"errors":[{"message":"Invalid token"}]}' | "${JQ}" -r '.errors[0].message // empty' 2>/dev/null && echo "PASS: jq error parse" || echo "FAIL: jq error parse"
echo '{"fields":[{"name":"DNS Key","value":"tok123"},{"name":"CF_Zone_ID","value":"zone456"}]}' | "${JQ}" -r '.fields[] | select(.name=="DNS Key") | .value' 2>/dev/null && echo "PASS: jq field select" || echo "FAIL: jq field select"

echo ""
echo "=== Test 3: Marker file creation ==="
MARKER_TEST="/tmp/.acme_marker_test"
cat > "${MARKER_TEST}" <<MEOF
domain=*.dtmventures.com
installed=$(date -Iseconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S')
acme_home=/share/CACHEDEV1_DATA/.acme.sh
email=test@test.com
MEOF
test -f "${MARKER_TEST}" && echo "PASS: marker written" || echo "FAIL: marker write"
cat "${MARKER_TEST}"
rm -f "${MARKER_TEST}"

echo ""
echo "=== Test 4: autorun.sh check ==="
test -f /etc/config/autorun.sh && echo "PASS: autorun.sh exists" || echo "FAIL: no autorun.sh"
grep -q "acme-cert-renew" /etc/config/autorun.sh 2>/dev/null && echo "INFO: acme already in autorun" || echo "INFO: acme NOT in autorun"

echo ""
echo "=== Test 5: Services ==="
test -x /etc/init.d/stunnel.sh && echo "PASS: stunnel.sh" || echo "INFO: no stunnel.sh"
test -x /etc/init.d/thttpd.sh && echo "PASS: thttpd.sh" || echo "INFO: no thttpd.sh"

echo ""
echo "=== Test 6: SSL directory ==="
ls -la /etc/config/ssl/ 2>/dev/null
test -d /etc/config/ssl && echo "PASS: ssl dir exists" || echo "FAIL: no ssl dir"
mkdir -p /etc/config/ssl 2>/dev/null && echo "PASS: ssl dir writable" || echo "INFO: ssl dir not writable (needs root)"

echo ""
echo "=== Test 7: Stunnel cert current subject ==="
openssl x509 -in /etc/stunnel/stunnel.pem -noout -subject -issuer -dates 2>/dev/null || echo "INFO: cannot read stunnel.pem (needs root)"

echo ""
echo "=== Test 8: acme.sh download test ==="
"${CURL}" -fsSL -o /dev/null --connect-timeout 10 "https://get.acme.sh" 2>/dev/null && echo "PASS: get.acme.sh reachable" || echo "FAIL: get.acme.sh unreachable"

echo ""
echo "=== Test 9: Cloudflare API connectivity ==="
# Note: /client/v4/ with no endpoint returns HTTP 400 "No route" — that's fine, it means CF is reachable
CF_HTTP="$("${CURL}" -sS -o /dev/null -w '%{http_code}' --connect-timeout 10 "https://api.cloudflare.com/client/v4/" 2>/dev/null)"
if [ "${CF_HTTP}" = "000" ]; then
    echo "FAIL: CF API unreachable (connection failed)"
else
    echo "PASS: CF API reachable (HTTP ${CF_HTTP})"
fi

echo ""
echo "=== Test 10: Bash version/features ==="
bash --version | head -1
echo "set -euo pipefail test:"
bash -c 'set -euo pipefail; echo "PASS: pipefail"' 2>/dev/null || echo "FAIL: pipefail"

echo ""
echo "=== ALL TESTS COMPLETE ==="
