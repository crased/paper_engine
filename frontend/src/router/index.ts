import { createRouter, createWebHistory } from "vue-router";

const routes = [
  { path: "/", name: "home", component: () => import("@/pages/HomePage.vue") },
  { path: "/metrics", name: "metrics", component: () => import("@/pages/MetricsPage.vue") },
  { path: "/test", name: "test", component: () => import("@/pages/TestPage.vue") },
  { path: "/tools", name: "tools", component: () => import("@/pages/ToolsPage.vue") },
  { path: "/settings", name: "settings", component: () => import("@/pages/SettingsPage.vue") },
];

export default createRouter({
  history: createWebHistory(),
  routes,
});
