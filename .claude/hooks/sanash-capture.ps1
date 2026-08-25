# sanash-capture.ps1 — SubagentStop hook.
# Best-effort capture of a subagent's final output into the Obsidian inbox.
# Design rule: NEVER break the session. Any failure -> exit 0 silently.
# Known limitation: agent type is inferred from the last Task tool_use in the
# transcript. Sequential subagents label correctly; parallel ones may mislabel.
# Nothing is lost either way — unmatched runs go to an "other-<date>.md" file.

$ErrorActionPreference = 'SilentlyContinue'

try {
    $raw = [Console]::In.ReadToEnd()
    if (-not $raw) { exit 0 }

    $payload = $raw | ConvertFrom-Json
    $tp = $payload.transcript_path
    if (-not $tp -or -not (Test-Path -LiteralPath $tp)) { exit 0 }

    $lines = Get-Content -LiteralPath $tp -Encoding UTF8
    $objs = New-Object System.Collections.ArrayList
    foreach ($ln in $lines) {
        if (-not $ln) { continue }
        try { [void]$objs.Add(($ln | ConvertFrom-Json)) } catch {}
    }
    if ($objs.Count -eq 0) { exit 0 }

    function Get-Text($msg) {
        if ($null -eq $msg) { return '' }
        $c = $msg.content
        if ($null -eq $c) { return '' }
        if ($c -is [string]) { return $c }
        $sb = New-Object System.Text.StringBuilder
        foreach ($b in $c) {
            if ($b.type -eq 'text' -and $b.text) { [void]$sb.AppendLine($b.text) }
        }
        return $sb.ToString()
    }

    # Last sidechain assistant message = the subagent's final report.
    $report = ''
    for ($i = $objs.Count - 1; $i -ge 0; $i--) {
        $o = $objs[$i]
        if ($o.type -eq 'assistant' -and $o.isSidechain -eq $true) {
            $t = Get-Text $o.message
            if ($t.Trim()) { $report = $t.Trim(); break }
        }
    }
    if (-not $report) { exit 0 }  # no subagent output found -> skip

    # Best-effort agent type + task label from the last main-chain Task tool_use.
    $agent = 'other'; $task = ''
    for ($i = $objs.Count - 1; $i -ge 0; $i--) {
        $o = $objs[$i]
        if ($o.isSidechain -eq $true) { continue }
        $c = $o.message.content
        if ($c -is [System.Array]) {
            $found = $false
            foreach ($b in $c) {
                if ($b.type -eq 'tool_use' -and $b.name -eq 'Task') {
                    if ($b.input.subagent_type) { $agent = "$($b.input.subagent_type)".ToLower() }
                    if ($b.input.description)   { $task  = "$($b.input.description)" }
                    $found = $true; break
                }
            }
            if ($found) { break }
        }
    }

    $known = @('donatello','danyshpan','cady','pitch')
    $slug = if ($known -contains $agent) { $agent } else { 'other' }
    if (-not $task) { $task = 'subagent run' }

    $dir = 'C:\Users\User\OneDrive\Desktop\obsidian\DiyasVault\00_Inbox\sanash-agents'
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    $date = Get-Date -Format 'yyyy-MM-dd'
    $time = Get-Date -Format 'HH:mm'
    $file = Join-Path $dir "$slug-$date.md"

    if (-not (Test-Path -LiteralPath $file)) {
        $fm = @"
---
title: $slug capture $date
tags:
  - agent-capture
  - $slug
created: $date
status: inbox
---

# $slug capture $date

Raw agent output captured automatically by the SubagentStop hook. This is a
throwaway staging file. Curate into clean project notes with
/sanash-to-obsidian, then this file moves to 06_Archive.

"@
        Set-Content -LiteralPath $file -Value $fm -Encoding UTF8
    }

    $entry = @"

## $time - $task

$report

---
"@
    Add-Content -LiteralPath $file -Value $entry -Encoding UTF8
    exit 0
}
catch {
    exit 0
}
