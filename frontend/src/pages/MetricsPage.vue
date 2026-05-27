<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { api } from "@/api/client";
import type { DatasetStats, ModelInfo } from "@/types";

const yamlChoice = ref("dataset_gameplay.yaml");
const stats = ref<DatasetStats | null>(null);
const models = ref<ModelInfo[]>([]);
const error = ref<string | null>(null);

const maxInstances = computed(() =>
  stats.value ? Math.max(1, ...stats.value.classes.map((c) => c.instances)) : 1,
);

async function load() {
  error.value = null;
  try {
    [stats.value, models.value] = await Promise.all([
      api.datasetStats(yamlChoice.value),
      api.models(),
    ]);
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  }
}

onMounted(load);

function fmtDate(epoch: number) {
  return new Date(epoch * 1000).toLocaleString();
}
</script>

<template>
  <h1>Metrics</h1>
  <p class="subtitle">Dataset balance and model inventory.</p>
  <p v-if="error" class="badge bad">{{ error }}</p>

  <div class="card">
    <div class="row" style="justify-content: space-between">
      <h2 style="margin: 0">Dataset class balance</h2>
      <div class="row">
        <select v-model="yamlChoice" @change="load">
          <option value="dataset.yaml">dataset.yaml (12-class)</option>
          <option value="dataset_gameplay.yaml">dataset_gameplay.yaml (4-class)</option>
        </select>
      </div>
    </div>

    <template v-if="stats">
      <div class="row muted" style="margin: 8px 0 14px; gap: 20px">
        <span>train: {{ stats.train_images }} imgs / {{ stats.train_labels }} labels</span>
        <span>val: {{ stats.val_images }} imgs / {{ stats.val_labels }} labels</span>
        <span>{{ stats.total_instances }} instances</span>
      </div>
      <div v-for="c in stats.classes" :key="c.cls_id" class="bar-row">
        <span class="bar-label">{{ c.cls_name }}</span>
        <div class="bar-track grow">
          <div class="bar-fill" :style="{ width: (c.instances / maxInstances) * 100 + '%' }" />
        </div>
        <span class="bar-count mono">{{ c.instances }}</span>
      </div>
    </template>
  </div>

  <div class="card">
    <h2 style="margin-top: 0">Trained models</h2>
    <table>
      <thead>
        <tr><th>Name</th><th>Size</th><th>Modified</th><th></th></tr>
      </thead>
      <tbody>
        <tr v-for="m in models" :key="m.weights_path">
          <td class="mono">{{ m.name }}</td>
          <td>{{ m.size_mb }} MB</td>
          <td class="muted">{{ fmtDate(m.modified) }}</td>
          <td><span v-if="m.active" class="badge accent">active</span></td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.bar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.bar-label { width: 110px; color: var(--text-1); font-size: 13px; }
.bar-count { width: 56px; text-align: right; color: var(--text-1); }
</style>
