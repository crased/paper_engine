<script setup lang="ts">
// Settings is a stub: the backend has no settings endpoints yet. The old GUI's
// SettingsPage managed LLM provider, API key, training packages, cloud storage,
// memory-reading config. Those map to future /api/settings routes.
const planned = [
  "LLM provider + API key (keyring-backed)",
  "Training params (epochs, model size, dataset yaml)",
  "Memory-reading config (host-only mode)",
  "Cloud storage / packages",
];
</script>

<template>
  <h1>Settings</h1>
  <p class="subtitle">Not wired yet — backend settings endpoints are pending.</p>

  <div class="card">
    <h2 style="margin-top: 0">Planned</h2>
    <ul>
      <li v-for="p in planned" :key="p" class="muted">{{ p }}</li>
    </ul>
    <p class="muted" style="margin-top: 12px">
      Next backend step: <span class="mono">GET/PUT /api/settings</span> reading
      <span class="mono">conf/*.ini</span> and the OS keyring.
    </p>
  </div>
</template>
