import { defineStore } from "pinia";
import { ref } from "vue";
import { api } from "@/api/client";
import type { ModelInfo } from "@/types";

export const useEngineStore = defineStore("engine", () => {
  const online = ref(false);
  const models = ref<ModelInfo[]>([]);
  const error = ref<string | null>(null);

  async function refresh() {
    error.value = null;
    try {
      await api.health();
      online.value = true;
      models.value = await api.models();
    } catch (e) {
      online.value = false;
      error.value = e instanceof Error ? e.message : String(e);
    }
  }

  return { online, models, error, refresh };
});
