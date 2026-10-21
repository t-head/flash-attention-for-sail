# Shared by the isolated Holmes wheel runners. No global Git configuration changes.
holmes_trust_checkout() {
  local checkout=$1 count=${GIT_CONFIG_COUNT:-0} directory
  # Scheduler tar extraction can preserve a different uid. Trust only this
  # checkout and its ACTLIZE dependency, for this process and its children.
  for directory in "$checkout" "$checkout/csrc/actlize"; do
    export "GIT_CONFIG_KEY_${count}=safe.directory"
    export "GIT_CONFIG_VALUE_${count}=$directory"
    count=$((count + 1))
  done
  export GIT_CONFIG_COUNT=$count
}
