#!/usr/bin/env bash

latentvis_parse_jobid() {
  local text="$1"
  sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' <<<"$text" | head -n 1
}

latentvis_needs_shared_reason() {
  local state_reason="$1"
  local patterns="${LATENTVIS_SHARED_FALLBACK_PATTERNS:-QOS|Assoc|Account|Billing|Grp|Limit}"
  [[ "$state_reason" =~ ^PENDING\| ]] && [[ "$state_reason" =~ $patterns ]]
}

latentvis_query_state_reason() {
  local jobid="$1"
  local tries="${LATENTVIS_QUEUE_CHECK_RETRIES:-6}"
  local sleep_s="${LATENTVIS_QUEUE_CHECK_SLEEP:-5}"
  local i
  local state_reason=""
  for i in $(seq 1 "$tries"); do
    state_reason="$(squeue -h -j "$jobid" -o "%T|%R" 2>/dev/null | head -n 1 || true)"
    if [[ -n "$state_reason" ]]; then
      printf '%s\n' "$state_reason"
      return 0
    fi
    sleep "$sleep_s"
  done
  return 1
}

latentvis_submit_variant() {
  local script_path="$1"
  local log_label="$2"
  local sbatch_args_name="$3"
  local script_args_name="$4"
  local -n _sbatch_args="$sbatch_args_name"
  local -n _script_args="$script_args_name"
  local output
  echo "[submit:$log_label] sbatch ${_sbatch_args[*]} $script_path ${_script_args[*]}"
  output="$(sbatch "${_sbatch_args[@]}" "$script_path" "${_script_args[@]}" 2>&1)"
  local rc=$?
  printf '%s\n' "$output"
  return "$rc"
}

latentvis_submit_shared_chain() {
  local shared_script="$1"
  local overrun_script="$2"
  local sbatch_args_name="$3"
  local script_args_name="$4"
  local -n _sbatch_args="$sbatch_args_name"
  local -n _script_args="$script_args_name"
  local rc

  set +e
  latentvis_submit_variant "$shared_script" "shared" "$sbatch_args_name" "$script_args_name"
  rc=$?
  set -e
  if [[ $rc -eq 0 || -z "$overrun_script" ]]; then
    return "$rc"
  fi

  echo "[submit:shared] shared submission failed; retrying on shared_overrun"
  latentvis_submit_variant "$overrun_script" "shared_overrun" "$sbatch_args_name" "$script_args_name"
}

latentvis_submit_with_fallback() {
  local regular_script="$1"
  local shared_script="$2"
  local overrun_script="$3"
  local sbatch_args_name="$4"
  local script_args_name="$5"
  local -n _sbatch_args="$sbatch_args_name"
  local -n _script_args="$script_args_name"
  local mode="${LATENTVIS_SUBMIT_MODE:-auto}"
  local output
  local rc
  local jobid
  local state_reason

  case "$mode" in
    regular)
      latentvis_submit_variant "$regular_script" "regular" "$sbatch_args_name" "$script_args_name"
      return $?
      ;;
    shared)
      latentvis_submit_shared_chain "$shared_script" "$overrun_script" sbatch_args_name script_args_name
      return $?
      ;;
    shared_overrun)
      latentvis_submit_variant "$overrun_script" "shared_overrun" "$sbatch_args_name" "$script_args_name"
      return $?
      ;;
    auto)
      ;;
    *)
      echo "[submit] invalid LATENTVIS_SUBMIT_MODE=$mode" >&2
      return 2
      ;;
  esac

  set +e
  output="$(latentvis_submit_variant "$regular_script" "regular" "$sbatch_args_name" "$script_args_name")"
  rc=$?
  set -e
  printf '%s\n' "$output"
  if [[ $rc -ne 0 ]]; then
    echo "[submit:auto] regular submission failed; retrying on shared"
    latentvis_submit_shared_chain "$shared_script" "$overrun_script" sbatch_args_name script_args_name
    return $?
  fi

  jobid="$(latentvis_parse_jobid "$output")"
  if [[ -z "$jobid" ]]; then
    echo "[submit:auto] could not parse regular job id; keeping regular submission"
    return 0
  fi

  state_reason="$(latentvis_query_state_reason "$jobid" || true)"
  if [[ -z "$state_reason" ]]; then
    echo "[submit:auto] queue state unavailable for job $jobid; keeping regular submission"
    return 0
  fi
  echo "[submit:auto] regular job $jobid state=$state_reason"
  if ! latentvis_needs_shared_reason "$state_reason"; then
    return 0
  fi

  echo "[submit:auto] switching job $jobid to shared because state=$state_reason"
  scancel "$jobid" >/dev/null 2>&1 || true
  latentvis_submit_shared_chain "$shared_script" "$overrun_script" sbatch_args_name script_args_name
}
