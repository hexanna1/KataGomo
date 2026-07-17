# Keep terminal output concise; retain complete stage output in the run's logs.
run_stage() {
    local stage="$1"
    shift
    local logfile="$DATA_DIR_ABS/logs/$stage.log"
    mkdir -p "$DATA_DIR_ABS/logs"
    echo "[$TRAINING_NAME] $stage"
    if "$@" 2>&1 | tee -a "$logfile" | awk '
        /: nsamp=/ || /Started [0-9]+ games/ || /Finished [0-9]+ games/ || /NN rows:/ {
            print; fflush(); next
        }
        /^p0loss = / {
            n = split($0, fields, ", ")
            summary = "validation: "
            for (i = 1; i <= n; i++)
                if (fields[i] ~ /^(p0loss|vloss|nsamp_train) = /)
                    summary = summary fields[i] "; "
            print summary; fflush(); next
        }
        tolower($0) ~ /(^|[^a-z])(warning|error|fatal|traceback)([^a-z]|$)/ {
            print; fflush()
        }
    '; then
        return 0
    else
        local status=$?
        echo "[$TRAINING_NAME] $stage failed (exit $status). Full log: $logfile" >&2
        tail -n 40 "$logfile" >&2
        return "$status"
    fi
}
