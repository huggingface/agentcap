# Demo: agentcap export pushes one or more runs to the Hub, then
# shows the resulting datasets via ``hf datasets info``.
#
# The tape pins ``AGENTCAP_WORKSPACE`` to a dedicated demo dir under
# ``$HOME`` so it only exports the demo's run (not whatever sweeps
# happen to be in the user's cwd). ``$HOME`` is the only path
# reliably shared into the podman VM on macOS — ``/tmp`` is NOT
# bind-mounted, so a ``/tmp/...`` workspace breaks the agent
# sandbox with "workdir does not exist on container".
#
# A hidden setup block wipes the demo dir and seeds a fresh run, so
# ``vhs docs/demo/export.tape`` is self-contained.
#
# Prerequisites:
#   - HF_TOKEN with write access to ``dacorvo/`` (swap the namespace
#     in the ``--push`` argument to push elsewhere).
#   - ``trufflehog`` on PATH (`brew install trufflehog` /
#     `apt install trufflehog`).
#   - podman running + the ``agentcap-pi`` image already built (one
#     prior ``agentcap run --agent pi …`` is enough).
#
# Regenerate the GIF from the repo root:
#   vhs docs/demo/export.tape

Output docs/img/export.gif

Set FontSize 14
Set Width 1400
Set Height 700
Set Theme "Catppuccin Mocha"
Set TypingSpeed 30ms

# Reset the prompt + wipe both the local workspace AND the
# destination datasets, then seed a fresh run. All hidden so the
# GIF only shows the export step. Re-running the tape always
# starts from a clean state in both halves of the pipeline, so the
# pushed dataset has exactly one parquet (not the union of every
# prior recording).
Hide
Type "export PS1='$ '"
Enter
Type "unset PROMPT_COMMAND"
Enter
Type "export AGENTCAP_WORKSPACE=$HOME/agentcap-demo-ws"
Enter
Type "rm -rf $AGENTCAP_WORKSPACE && mkdir -p $AGENTCAP_WORKSPACE"
Enter
# ``--missing-ok`` so the first-ever run doesn't fail when the
# repos don't exist yet; subsequent runs find them, delete them,
# and let ``agentcap export`` re-create them.
Type "hf repos delete -y --missing-ok --type dataset dacorvo/agentcap-demo-captures"
Enter
Type "hf repos delete -y --missing-ok --type dataset dacorvo/agentcap-demo-pi-traces"
Enter
Type "agentcap run --agent pi --model zai-org/GLM-4.6 \"
Enter
Type "    --upstream https://router.huggingface.co \"
Enter
Type "    --tasks docs/demo/tasks.txt --turns 2 --followup synthesized"
Enter
# Verified: pi/GLM-4.6/2 turns lands in ~12s on a warm image; the
# 60s buffer covers a cold image or a slower router response.
Sleep 60s
Type "clear"
Enter
Show

# Push every run in the demo workspace under one ``--push <owner>/<base>``
# value — drives captures, per-agent traces, and the collection in
# one command.
Type "agentcap export --all --push dacorvo/agentcap-demo"
Enter

# trufflehog scan + upload. Verified: ~12-row run pushes in ~15s.
# Bump if the workspace is large or the upstream is slow; the
# per-row progress bars give natural visual feedback meanwhile.
Sleep 30s

# Show the captures repo metadata.
Type "hf datasets info dacorvo/agentcap-demo-captures"
Enter
Sleep 5s

# Per-agent traces land in their own repo. The seed agent is pi.
Type "hf datasets info dacorvo/agentcap-demo-pi-traces"
Enter
Sleep 5s
