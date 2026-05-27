<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { api } from "@/api/client";
import type { InferResponse, ModelInfo } from "@/types";

const models = ref<ModelInfo[]>([]);
const sessions = ref<string[]>([]);

const model = ref<string>("");
const session = ref<string>("");
const directory = ref<string>("");
const conf = ref(0.25);
const limit = ref(16);

const running = ref(false);
const error = ref<string | null>(null);
const result = ref<InferResponse | null>(null);

const classDist = computed(() => {
  if (!result.value) return [];
  const counts: Record<string, number> = {};
  for (const f of result.value.frames)
    for (const d of f.detections) counts[d.cls_name] = (counts[d.cls_name] || 0) + 1;
  return Object.entries(counts).sort((a, b) => b[1] - a[1]);
});

onMounted(async () => {
  [models.value, sessions.value] = await Promise.all([api.models(), api.sessions()]);
  model.value = models.value.find((m) => m.active)?.name ?? models.value[0]?.name ?? "";
  session.value = sessions.value[0] ?? "";
});

async function run() {
  running.value = true;
  error.value = null;
  result.value = null;
  try {
    result.value = await api.infer({
      model: model.value || null,
      session: directory.value ? null : session.value || null,
      directory: directory.value || null,
      conf: conf.value,
      limit: limit.value,
    });
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    running.value = false;
  }
}

function baseName(p: string) {
  return p.split("/").pop() ?? p;
}
</script>

<template>
  <h1>Test — inference</h1>
  <p class="subtitle">Run a trained model over a session or directory and inspect detections.</p>

  <div class="card">
    <div class="row">
      <div>
        <label>Model</label>
        <select v-model="model">
          <option v-for="m in models" :key="m.name" :value="m.name">
            {{ m.name }}{{ m.active ? " (active)" : "" }}
          </option>
        </select>
      </div>
      <div>
        <label>Session</label>
        <select v-model="session" :disabled="!!directory">
          <option v-for="s in sessions" :key="s" :value="s">{{ s }}</option>
        </select>
      </div>
      <div class="grow">
        <label>…or directory (absolute path)</label>
        <input v-model="directory" placeholder="/path/to/images" class="grow" style="width: 100%" />
      </div>
    </div>
    <div class="row" style="margin-top: 12px">
      <div>
        <label>Confidence: {{ conf.toFixed(2) }}</label>
        <input type="range" min="0.05" max="0.9" step="0.05" v-model.number="conf" />
      </div>
      <div>
        <label>Frame limit</label>
        <input type="number" v-model.number="limit" min="0" max="500" style="width: 90px" />
      </div>
      <div style="align-self: flex-end">
        <button :disabled="running" @click="run">{{ running ? "Running…" : "Run inference" }}</button>
      </div>
    </div>
    <p v-if="error" class="badge bad" style="margin-top: 12px">{{ error }}</p>
  </div>

  <div v-if="result" class="card">
    <div class="row" style="gap: 20px">
      <span class="badge accent">{{ result.frames_with_detection }}/{{ result.frames.length }} frames hit</span>
      <span class="muted">{{ result.total_detections }} detections</span>
      <span class="muted mono">{{ baseName(result.model) }}</span>
    </div>
    <div class="row" style="margin-top: 12px; gap: 8px">
      <span v-for="[name, n] in classDist" :key="name" class="badge accent">{{ name }}: {{ n }}</span>
      <span v-if="!classDist.length" class="muted">no detections</span>
    </div>

    <h2>Per-frame</h2>
    <table>
      <thead>
        <tr><th>Frame</th><th>#</th><th>Detections (conf)</th></tr>
      </thead>
      <tbody>
        <tr v-for="f in result.frames" :key="f.image">
          <td class="mono">{{ baseName(f.image) }}</td>
          <td>{{ f.detections.length }}</td>
          <td>
            <span v-if="f.error" class="badge bad">{{ f.error }}</span>
            <span v-else class="muted">
              {{ f.detections.map((d) => `${d.cls_name}:${d.confidence.toFixed(2)}`).join(", ") || "—" }}
            </span>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
