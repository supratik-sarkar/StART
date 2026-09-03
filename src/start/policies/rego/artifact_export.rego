package start.export.artifact_filtering

import rego.v1

default allow = false

# Allow artifact export only if it does not contain raw client datasets
allow if {
    input.contains_raw_dataset == false
}

reason = sprintf("Artifact '%v' (%v) contains only sanitized metrics and is approved for export.", [input.artifact_id, input.artifact_type]) if {
    allow
} else = sprintf("Artifact '%v' contains unsanitized raw datasets and is blocked by data leak prevention policy.", [input.artifact_id])
