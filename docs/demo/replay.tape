# Demo: agentcap replay re-issues one captured request from the HF
# dataset against HF Router. Re-runs the exact body the agent sent
# the first time; differences in output come from the target, not
# from agentcap normalising the payload.
#
# Prerequisites:
#   - HF_TOKEN exported, or ~/.cache/huggingface/token via ``hf auth login``.
#
# The rid below is one turn-1 / task_01 call from the pi parquet in
# dacorvo/hf-hub-session-captures — picked for brevity so the GIF
# doesn't sit through a long generation. Swap it out (e.g. via
# ``agentcap inspect --rid --source dacorvo/hf-hub-session-captures``)
# if you want to demo a different call.
#
# Regenerate the GIF from the repo root:
#   vhs docs/demo/replay.tape

Output docs/img/replay.gif

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

# Type the replay command line by line so the GIF reads cleanly.
Type "agentcap replay 51376d58584d498bbf9fa0d3efaba85c \"
Enter
Type "    --target https://router.huggingface.co \"
Enter
Type "    --source dacorvo/hf-hub-session-captures"
Enter

# Watch the streamed response come back. HF Router latency on a
# short turn-1 message lands in 5–20s; bump if you swapped in a
# longer rid.
Sleep 30s
