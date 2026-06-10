# Demo: agentcap inspect drilling into an HF dataset of captures.
#
# Uses dacorvo/hf-hub-session-captures as the source.
#
# Regenerate the GIF from the repo root:
#   vhs docs/demo/inspect.tape

Output docs/img/inspect.gif

Set FontSize 14
Set Width 1400
Set Height 700
Set Theme "Catppuccin Mocha"
Set TypingSpeed 30ms

# Reset the shell prompt to a plain ``$`` and drop ``PROMPT_COMMAND``
# so a user's custom bash setup doesn't leak into the GIF.
Hide
Type "export PS1='$ '"
Enter
Type "unset PROMPT_COMMAND"
Enter
Type "clear"
Enter
Show

# Open the three-level picker chain over the HF dataset.
Type "agentcap inspect dacorvo/hf-hub-session-captures"
Enter
Sleep 8s

# Level 1 (parquet picker): narrow to hermes' run, look at the
# task list in the preview pane, then pick it.
Type "hermes"
Sleep 2s
Enter
Sleep 3s

# Level 2 (request picker): search across deep new-message content
# — ``hf-cli`` matches rows whose diff references the skill name,
# not just the visible summary column.
Type "hf-cli"
Sleep 3s

# Walk down a few rows to show the preview pane updating live.
Down
Sleep 600ms
Down
Sleep 600ms
Down
Sleep 600ms

# Drill into the picked request's flattened conversation.
Enter
Sleep 3s

# Level 3 (message picker): browse a couple of messages — the
# preview shows full content for the highlighted role.
Down
Sleep 700ms
Down
Sleep 700ms
Down
Sleep 700ms

# Esc walks back one level at a time. Three Escs to exit cleanly.
Escape
Sleep 800ms
Escape
Sleep 800ms
Escape
Sleep 1s
