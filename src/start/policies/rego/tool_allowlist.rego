package start.tools.execution_allowlist

import rego.v1

default allow = false

# Allow tool execution if tool is present in authorized allowlist
allow if {
    input.tool_name in input.allowlist
}

reason = sprintf("Tool '%v' is authorized for execution.", [input.tool_name]) if {
    allow
} else = sprintf("Tool '%v' is not in the authorized validation tool registry.", [input.tool_name])
