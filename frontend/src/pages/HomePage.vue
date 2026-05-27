<script setup lang="ts">
import { storeToRefs } from "pinia";
import { useEngineStore } from "@/stores/engine";

const engine = useEngineStore();
const { online, models, error } = storeToRefs(engine);

const cards = [
  { to: "/test", title: "Run inference", desc: "Point a trained model at frames and see detections." },
  { to: "/tools", title: "Train / annotate", desc: "Launch jobs, stream live logs, read reports." },
  { to: "/metrics", title: "Metrics", desc: "Dataset class balance and model inventory." },
];
</script>

<template>
  <h1>Paper Engine</h1>
  <p class="subtitle">Game object detection — localhost control plane.</p>

  <div class="card">
    <div class="row">
      <span class="badge" :class="online ? 'good' : 'bad'">
        {{ online ? "engine online" : "engine offline" }}
      </span>
      <span class="muted">{{ models.length }} trained model(s)</span>
      <button class="ghost" @click="engine.refresh">Refresh</button>
    </div>
    <p v-if="error" class="muted" style="margin-top: 10px">{{ error }}</p>
  </div>

  <div class="row">
    <RouterLink v-for="c in cards" :key="c.to" :to="c.to" class="card grow quick">
      <strong>{{ c.title }}</strong>
      <span class="muted">{{ c.desc }}</span>
    </RouterLink>
  </div>
</template>

<style scoped>
.quick { display: flex; flex-direction: column; gap: 6px; min-width: 220px; }
.quick strong { color: var(--text-0); }
.quick:hover { border-color: var(--accent); }
</style>
