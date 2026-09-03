package start.governance.attestation_rules

import rego.v1

default allow = false

# Allow governance sign-off if zero ungrounded claims and valid committee disposition
allow if {
    input.n_ungrounded_claims == 0
    input.committee_disposition in ["ACCEPT", "ACCEPT_WITH_CONDITIONS", "REMEDIATION_REQUIRED"]
    not (input.n_validation_failures > 0 and input.committee_disposition == "ACCEPT")
}

reason = sprintf("Governance attestation criteria satisfied (disposition: %v).", [input.committee_disposition]) if {
    allow
} else = "Governance attestation denied: ungrounded claims or invalid unconditional accept."
