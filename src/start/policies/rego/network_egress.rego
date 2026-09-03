package start.security.network_egress

import rego.v1

default allow = false

# Allow local network connection
allow if {
    input.target_host in ["localhost", "127.0.0.1", "in-memory"]
}

# Allow external egress only when private zero-egress mode is false
allow if {
    input.private_mode == false
}

reason = "Local connection or explicit egress permitted." if {
    allow
} else = "External network egress is blocked by zero-egress policy."
