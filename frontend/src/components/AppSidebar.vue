<script setup lang="ts">
import { onMounted } from "vue";
import { storeToRefs } from "pinia";
import { useEngineStore } from "@/stores/engine";

const nav = [
  { to: "/", icon: "◆", label: "Home" },
  { to: "/metrics", icon: "▣", label: "Metrics" },
  { to: "/test", icon: "▶", label: "Test" },
  { to: "/tools", icon: "✦", label: "Tools" },
  { to: "/settings", icon: "⚙", label: "Settings" },
];

const engine = useEngineStore();
const { online } = storeToRefs(engine);
onMounted(engine.refresh);
</script>

<template>
  <aside class="sidebar">
    <div class="brand">Paper Engine</div>
    <nav>
      <RouterLink v-for="item in nav" :key="item.to" :to="item.to" class="nav-item">
        <span class="icon">{{ item.icon }}</span>{{ item.label }}
      </RouterLink>
    </nav>
    <div class="status">
      <span class="dot" :class="online ? 'on' : 'off'" />
      {{ online ? "engine online" : "engine offline" }}
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 156px;
  background: var(--bg-1);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 16px 12px;
}
.brand {
  font-weight: 700;
  font-size: 15px;
  padding: 6px 8px 18px;
  color: var(--text-0);
}
nav { display: flex; flex-direction: column; gap: 2px; flex: 1; }
.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 10px;
  border-radius: 7px;
  color: var(--text-1);
  font-size: 13px;
  font-weight: 500;
}
.nav-item:hover { background: var(--bg-2); color: var(--text-0); }
.nav-item.router-link-active { background: var(--accent-dim); color: #9ec5ff; }
.icon { width: 16px; text-align: center; }
.status {
  font-size: 11px;
  color: var(--text-2);
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 8px;
}
.dot { width: 8px; height: 8px; border-radius: 50%; }
.dot.on { background: var(--good); }
.dot.off { background: var(--bad); }
</style>
