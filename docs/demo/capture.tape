# Demo: agentcap run captures one task end-to-end against HF Router.
#
# Picked HF Router as the upstream to keep the demo's prereqs small —
# no local model server to spawn, no GGUF download. Only requirement
# beyond agentcap + podman is an HF_TOKEN with router access.
#
# Prerequisites:
#   - podman running and the per-agent image already built (one cold
#     run is enough to populate the cache before recording).
#   - HF_TOKEN exported, or ~/.cache/huggingface/token written via
#     ``hf auth login``.
#
# The tape drives off docs/demo/tasks.txt (one short prompt, versioned
# alongside this file) so the recording is self-contained — no
# external file to seed before running vhs.
#
# Regenerate the GIF from the repo root:
#   vhs docs/demo/capture.tape

Output docs/img/capture.gif

Set FontSize 14
Set Width 1400
Set Height 700
Set Theme "Catppuccin Mocha"
Set TypingSpeed 30ms

# Reset the shell prompt to a plain ``$`` and drop ``PROMPT_COMMAND``
# so a user's custom bash setup (git_prompt_info hooks, virtualenv
# markers, …) doesn't leak into the GIF and break rendering on a
# different machine. Hidden so the GIF starts on a clean screen.
Hide
Type "export PS1='$ '"
Enter
Type "unset PROMPT_COMMAND"
Enter
Type "clear"
Enter
Show

# Type the agentcap run command line by line so the GIF reads cleanly.
Type "agentcap run \"
Enter
Type "    --agent pi \"
Enter
Type "    --model zai-org/GLM-4.6 \"
Enter
Type "    --upstream https://router.huggingface.co \"
Enter
Type "    --tasks docs/demo/tasks.txt \"
Enter
Type "    --turns 2 \"
Enter
Type "    --followup synthesized"
Enter

# Show the full run live — pi + GLM-4.6 over HF Router lands in
# ~30s for two short turns. The per-run summary line at the end
# already prints the capture path; no fake spinner needed.
Sleep 45s
