import json
from typing import Dict, List, Tuple, Literal
import pandas as pd
import numpy as np
import gradio as gr
from openai import OpenAI
from huggingface_hub import InferenceClient
from particleanalyzer.core.language_context import LanguageContext, resolve_language
from particleanalyzer.core.languages import translations


class LLMAnalysis:
    def __init__(
        self,
        api_key: str,
        provider: Literal["openrouter", "huggingface"] = "openrouter",
    ):
        self.provider = provider
        self.api_key = api_key

        if api_key and self.api_key.startswith("hf_"):
            provider == "huggingface"
            self.client = InferenceClient(provider="fireworks-ai", api_key=api_key)
            self.model_list = ["deepseek-ai/DeepSeek-V3"]
        elif api_key and self.api_key.startswith("sk-or-"):
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            self.model_list = [
                "google/gemini-2.0-flash-001",
                "deepseek/deepseek-chat-v3-0324",
                "deepseek/deepseek-r1-distill-llama-70b",
                "deepseek/deepseek-r1-0528:free",
                "openai/gpt-4o-mini",
            ]
            provider == "openrouter"
        elif api_key:
            raise ValueError(
                "Неизвестный провайдер. Доступные варианты: 'openrouter', 'huggingface'"
            )
        else:
            self.model_list = [None]

    def _calculate_stats(self, df: pd.DataFrame, num_bins: int = 5) -> Dict[str, Dict]:
        """Вычисляет статистику"""
        stats = {"particles_count": len(df), "parameters": {}}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            stats["parameters"][col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "q1": df[col].quantile(0.25),
                "q3": df[col].quantile(0.75),
                "skewness": df[col].skew(),
                "kurtosis": df[col].kurtosis(),
                "histogram": self._create_histogram(df[col], num_bins),
            }

            if "Dₘₐₓ" in col and any("Dₘᵢₙ" in c for c in df.columns):
                stats["parameters"]["aspect_ratio"] = self._calc_aspect_ratio(
                    df, num_bins
                )

            if "P [" in col and any("S [" in c for c in df.columns):
                stats["parameters"]["circularity"] = self._calc_circularity(
                    df, num_bins
                )
        return stats

    def _create_histogram(self, data, num_bins):
        """Создает гистограмму"""
        counts, bins = np.histogram(data, bins=num_bins)
        return {"bins": [float(x) for x in bins], "counts": [int(x) for x in counts]}

    def _calc_aspect_ratio(self, df, num_bins):
        """Вычисляет аспектное соотношение"""
        if "Nanorod aspect ratio (L/W)" in df.columns:
            ar = df["Nanorod aspect ratio (L/W)"]
        else:
            dmax_col = [c for c in df.columns if "Dₘₐₓ" in c][0]
            dmin_col = [c for c in df.columns if "Dₘᵢₙ" in c][0]
            ar = df[dmax_col] / df[dmin_col]
        return {
            "mean": float(ar.mean()),
            "median": float(ar.median()),
            "histogram": self._create_histogram(ar, num_bins),
        }

    def _calc_circularity(self, df, num_bins):
        """Вычисляет округлость"""
        p_col = [c for c in df.columns if "P [" in c][0]
        s_col = [c for c in df.columns if "S [" in c][0]
        circ = 4 * np.pi * df[s_col] / (df[p_col] ** 2)
        return {
            "mean": float(circ.mean()),
            "median": float(circ.median()),
            "histogram": self._create_histogram(circ, num_bins),
        }

    def analyze(
        self,
        df: pd.DataFrame,
        model_llm: str,
        request: gr.Request | None = None,
        selected_language: str = "auto",
    ) -> List[Tuple[None, str]]:
        """Анализирует DataFrame с частицами и возвращает результаты LLM"""
        lang = resolve_language(
            selected_language,
            request.headers.get("Accept-Language", "")
            if request is not None
            else "",
            fallback="en",
        )
        LanguageContext.set_language(lang)

        if df.empty:
            return [
                {
                    "role": "assistant",
                    "content": translations[lang]["No particles detected"],
                }
            ]

        # Вычисляем статистику
        stats = self._calculate_stats(df)
        count_particles = len(df)

        try:
            prompt = self._build_prompt(stats, count_particles, lang)
            self.model = model_llm
            response = self._get_llm_response(prompt)
            return self._format_response(response)
        except Exception as e:
            print(f"LLM analysis failed: {str(e)}")
            error_label = translations[lang]["Analysis error"]
            return [
                {"role": "assistant", "content": f"{error_label}: {str(e)}"}
            ]

    def _build_prompt(self, stats, count_particles, lang) -> str:
        return f"""
        Ты — ведущий эксперт в материаловедении и сканирующей электронной микроскопии (СЭМ) с 15-летним стажем. 
        Твоя задача — подготовить развёрнутый экспертный отчёт по характеристикам {count_particles} частиц.

        📌 **Контекст**:
        - Данные представлены в агрегированном виде (средние значения, медианы, стандартные отклонения и др.).
        - Размеры могут быть в микрометрах, нанометрах или пикселях.
        - Возможны аномально низкие значения из-за обрезки частиц по краям изображения. 
        - **Важно**: Крупные частицы считаются достоверными — **не упоминай влияние обрезки в их отношении**.
        - Не пересказывай входные данные — **только интерпретируй**
        - Опирайся на количественные показатели и сравнение между параметрами. Избегай общих фраз без числовой поддержки. Анализ должен быть научно обоснованным и лаконичным.

        🔬 **Задачи анализа**:
        1. **Распределение размеров** — тип, фракции, аномалии, полидисперсность  
        2. **Морфология** — форма, вытянутость, однородность  
        3. **Ориентация** — направленность
        4. **Проблемные зоны** — участки с сомнительными измерениями

        ✍️ **Формат отчёта (язык: {lang})** — без лишних отступов, используй маркированные списки и смайлики-иконки:
        
        🌡️ **Размерные характеристики**:
        - **Размер**: <диаметр, диапазоны, преобладающие фракции>  
        - **Тип распределения**: <нормальное / бимодальное / асимметричное>  
        - **Аномалии**: <выбросы, артефакты, влияние обрезки>  
        - **Полидисперсность**: <низкая / умеренная / высока

        🔵 **Морфология**:
        - **Преобладающая форма**: <основано на circularity, aspect_ratio, эксцентриситете e>  
        - **Однородность**: <оценка схожести форм>  
        - **Особенности**: <необычные формы, группы, дефекты>  

        🧭 **Ориентация**:
        - **Тип распределения углов**: <равномерное / направленное>  
        - **Предпочтительные направления**: <если есть>  

        ⚠️ **Проблемные зоны**:
        - <опиши участки с сомнительными или недостоверными измерениями>  
        
        📊 Выводы:
        - Используй числовые значения с точностью до 2 знаков после запятой  
        - Выводы должны быть лаконичными и научно обоснованными  
        - Рекомендации не требуются

        📁 **Данные для анализа** (агрегированные показатели):
        {json.dumps(stats, indent=2)}
        """

    def _get_llm_response(self, prompt: str):
        """Отправка запроса в LLM"""
        if self.provider == "openrouter":
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://sem.rybakov-k.ru/",
                    "X-Title": "ParticleAnalyzer",
                },
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return completion
        elif self.provider == "huggingface":
            return self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )

    def _format_response(self, response) -> List[Tuple[None, str]]:
        """Форматирование ответа LLM"""
        if self.provider == "openrouter":
            analysis = response.choices[0].message.content
        elif self.provider == "huggingface":
            analysis = response.choices[0].message.content
        return [{"role": "assistant", "content": analysis}]
