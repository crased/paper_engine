<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { api } from "@/api/client";
import type { JobInfo, ReportInfo } from "@/types";

// --- Reports ---
const reports = ref<ReportInfo[]>([]);
const reportName = ref<string>("");
const reportText = ref<string>("");

async function loadReports() {
  reports.value = await api.reports();
}
async function openReport(name: string) {
  reportName.value = name;
  reportText.value = (await api.report(name)).content;
}

// --- Jobs ---
const sessions = ref<string[]>([]);
const annSession = ref<string>("");
const annDataset = ref<string>("");
const annMaxFrames = ref(8);
const annNoExamples = ref(true);
const annDryRun = ref(false);

const log = ref<string[]>([]);
const activeJob = ref<JobInfo | null>(null);
let es: EventSource | null = null;

function attach(job: JobInfo) {
  activeJob.value = job;
  log.value = [];
  es?.close();
  es = api.streamJob(job.id);
  es.onmessage = (ev) => {
    log.value.push(ev.data);
  };
  es.addEventListener("end", (ev) => {
    log.value.push(`— job ${(ev as MessageEvent).data} —`);
    if (activeJob.value) activeJob.value.status = (ev as MessageEvent).data as JobInfo["status"];
    es?.close();
    es = null;
  });
  es.onerror = () => {
    es?.close();
    es = null;
  };
}

async function runTrain() {
  attach(await api.startTrain());
}
async function runAnnotate() {
  if (!annSession.value) return;
  attach(
    await api.startAnnotate({
      session: annSession.value,
      dataset: annDataset.value || null,
      max_frames: annMaxFrames.value,
      no_examples: annNoExamples.value,
      dry_run: annDryRun.value,
    }),
  );
}
async function cancel() {
  if (activeJob.value) await api.cancelJob(activeJob.value.id);
}

onMounted(async () => {
  await loadReports();
  sessions.value = await api.sessions();
  annSession.value = sessions.value[0] ?? "";
});
onUnmounted(() => es?.close());
</script>

<template>
  <h1>Tools</h1>
  <p class="subtitle">Run pipeline jobs and read generated reports.</p>

  <div class="card">
    <h2 style="margin-top: 0">Jobs</h2>
    <div class="row">
      <button @click="runTrain">Train (uses training_conf.ini)</button>
    </div>

    <h2>Annotate a session</h2>
    <div class="row">
      <div>
        <label>Session</label>
        <select v-model="annSession">
          <option v-for="s in sessions" :key="s" :value="s">{{ s }}</option>
        </select>
      </div>
      <div>
        <label>Dataset dir (optional)</label>
        <input v-model="annDataset" placeholder="yolo_dataset" />
      </div>
      <div>
        <label>Max frames</label>
        <input type="number" v-model.number="annMaxFrames" min="0" style="width: 84px" />
      </div>
      <label class="check"><input type="checkbox" v-model="annNoExamples" /> no few-shot</label>
      <label class="check"><input type="checkbox" v-model="annDryRun" /> dry run</label>
      <div style="align-self: flex-end">
        <button :disabled="!annSession" @click="runAnnotate">Annotate</button>
      </div>
    </div>

    <template v-if="activeJob">
      <div class="row" style="margin-top: 14px; justify-content: space-between">
        <span>
          <span class="badge" :class="{ good: activeJob.status === 'done', bad: activeJob.status === 'error', warn: activeJob.status === 'running' }">
            {{ activeJob.status }}
          </span>
          <span class="muted mono" style="margin-left: 8px">{{ activeJob.kind }} · {{ activeJob.id }}</span>
        </span>
        <button v-if="activeJob.status === 'running'" class="ghost" @click="cancel">Cancel</button>
      </div>
      <pre class="log">{{ log.join("\n") || "waiting for output…" }}</pre>
    </template>
  </div>

  <div class="card">
    <h2 style="margin-top: 0">Reports</h2>
    <div class="row" style="align-items: flex-start">
      <div style="min-width: 280px">
        <button class="ghost" @click="loadReports">Refresh</button>
        <ul class="report-list">
          <li v-for="r in reports" :key="r.name" :class="{ sel: r.name === reportName }" @click="openReport(r.name)">
            <span class="mono">{{ r.name }}</span>
          </li>
          <li v-if="!reports.length" class="muted">no reports yet</li>
        </ul>
      </div>
      <pre v-if="reportText" class="log grow">{{ reportText }}</pre>
    </div>
  </div>
</template>

<style scoped>
.check { display: inline-flex; align-items: center; gap: 6px; color: var(--text-1); font-size: 12px; align-self: flex-end; }
.report-list { list-style: none; padding: 0; margin: 10px 0 0; }
.report-list li { padding: 6px 8px; border-radius: 6px; cursor: pointer; font-size: 12px; }
.report-list li:hover { background: var(--bg-2); }
.report-list li.sel { background: var(--accent-dim); }
</style>
