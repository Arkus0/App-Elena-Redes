"""
AI Service - Grok-powered content analysis and generation
The brain behind high-engagement content creation
"""
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class AIService:
    """
    Service for AI-powered content analysis and generation using Grok (xAI)
    Specialized prompts for maximum engagement in local SMB content
    """

    def __init__(self):
        self.client = None
        self.model = settings.GROK_MODEL
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client for Grok"""
        if settings.GROK_API_KEY:
            try:
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(
                    api_key=settings.GROK_API_KEY,
                    base_url="https://api.x.ai/v1"
                )
                logger.info("Grok (OpenAI compatible) client initialized")
            except ImportError:
                logger.warning("openai package not installed")
        else:
            logger.warning("GROK_API_KEY not set, using mock responses")

    def is_available(self) -> bool:
        return self.client is not None

    async def analyze_posts_for_patterns(
        self,
        posts: List[Dict[str, Any]],
        business_type: str,
        platform: str
    ) -> Dict[str, Any]:
        """
        Analyze competitor posts to extract winning patterns
        Returns detailed pattern analysis for content generation
        """
        if not self.is_available():
            return self._get_mock_pattern_analysis(business_type, platform)

        system_prompt = self._get_pattern_analysis_system_prompt()
        user_prompt = self._build_pattern_analysis_prompt(posts, business_type, platform)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000
            )

            # Parse JSON response
            response_text = response.choices[0].message.content
            # Extract JSON from response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
            else:
                json_str = response_text

            return json.loads(json_str.strip())

        except Exception as e:
            logger.error(f"Error analyzing posts: {e}")
            return self._get_mock_pattern_analysis(business_type, platform)

    async def generate_content_piece(
        self,
        business_info: Dict[str, Any],
        patterns: List[Dict[str, Any]],
        platform: str,
        content_format: str,
        goal: str,
        similar_top_posts: List[Dict[str, Any]] = None,
        ml_recommendations: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a single high-engagement content piece
        Based on proven patterns and top-performing competitor posts
        Now enhanced with ML recommendations from the hybrid architecture
        """
        if not self.is_available():
            return self._get_mock_generated_content(business_info, platform, content_format, goal)

        system_prompt = self._get_content_generation_system_prompt(platform)
        user_prompt = self._build_content_generation_prompt(
            business_info, patterns, platform, content_format, goal, similar_top_posts,
            ml_recommendations
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=3000
            )

            response_text = response.choices[0].message.content
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
            else:
                json_str = response_text

            return json.loads(json_str.strip())

        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return self._get_mock_generated_content(business_info, platform, content_format, goal)

    async def predict_engagement(
        self,
        content: Dict[str, Any],
        patterns: List[Dict[str, Any]],
        competitor_benchmarks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict engagement score and explain why
        Returns score 0-100 with detailed explanation in Spanish
        """
        if not self.is_available():
            return self._get_mock_engagement_prediction(content)

        system_prompt = """Eres un experto en marketing de redes sociales para negocios locales.
        Tu tarea es predecir el engagement de contenido basandote en patrones probados.
        Responde SOLO en JSON con el formato especificado."""

        user_prompt = f"""Analiza este contenido y predice su engagement:

CONTENIDO:
{json.dumps(content, ensure_ascii=False, indent=2)}

PATRONES EXITOSOS DEL NICHO:
{json.dumps(patterns[:5], ensure_ascii=False, indent=2)}

BENCHMARKS DE COMPETIDORES:
{json.dumps(competitor_benchmarks, ensure_ascii=False, indent=2)}

Responde en JSON:
{{
    "score": 0-100,
    "confidence": 0-100,
    "breakdown": {{
        "hook_score": 0-100,
        "cta_score": 0-100,
        "format_score": 0-100,
        "trend_alignment": 0-100,
        "emotional_triggers": 0-100
    }},
    "explanation": "Explicacion en español de por qué este contenido funcionará (o no). Menciona posts similares de competidores que tuvieron exito.",
    "strengths": ["lista de fortalezas"],
    "improvements": ["sugerencias de mejora"],
    "predicted_metrics": {{
        "likes_range": "500-1000",
        "comments_range": "50-100",
        "saves_estimate": "200-400"
    }}
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500
            )

            response_text = response.choices[0].message.content
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            else:
                json_str = response_text

            return json.loads(json_str.strip())

        except Exception as e:
            logger.error(f"Error predicting engagement: {e}")
            return self._get_mock_engagement_prediction(content)

    async def generate_viral_ideas(
        self,
        trend_data: List[Dict[str, Any]],
        business_info: Dict[str, Any],
        keyword: str
    ) -> List[Dict[str, Any]]:
        """
        Generate reactive content ideas based on trending content
        """
        if not self.is_available():
            return self._get_mock_viral_ideas(keyword, business_info)

        system_prompt = """Eres un experto viral en TikTok e Instagram para negocios locales.
        Tu trabajo es adaptar tendencias virales al contexto de negocios pequeños.
        El contenido debe ser filmable con un telefono en menos de 1 hora."""

        user_prompt = f"""Basandote en estos videos/posts trending sobre "{keyword}":

{json.dumps(trend_data[:10], ensure_ascii=False, indent=2)}

Y este negocio:
- Tipo: {business_info.get('business_type')}
- Nombre: {business_info.get('name')}
- Ubicacion: {business_info.get('location')}

Genera 5 ideas de contenido viral adaptadas a este negocio.

Responde en JSON:
[
    {{
        "idea_title": "titulo corto",
        "platform": "instagram o tiktok",
        "format": "reel, tiktok_video, carousel",
        "hook": "los primeros 3 segundos (texto que aparece o se dice)",
        "script_outline": "estructura del video en bullet points",
        "filming_tips": "como filmarlo facilmente",
        "why_it_will_work": "explicacion",
        "difficulty": "facil/medio/dificil",
        "time_to_create": "30min, 1hora, etc",
        "trending_audio_suggestion": "nombre de audio trending o 'audio original'",
        "hashtags": ["lista", "de", "hashtags"]
    }}
]"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2500
            )

            response_text = response.choices[0].message.content
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            else:
                json_str = response_text

            return json.loads(json_str.strip())

        except Exception as e:
            logger.error(f"Error generating viral ideas: {e}")
            return self._get_mock_viral_ideas(keyword, business_info)

    # ============ SYSTEM PROMPTS ============

    def _get_pattern_analysis_system_prompt(self) -> str:
        return """Eres un experto analista de contenido viral en Instagram y TikTok especializado en negocios locales (floristerias, inmobiliarias, cafeterias, peluquerias, tiendas).

Tu mision es extraer PATRONES EXACTOS de posts exitosos que puedan replicarse.

IMPORTANTE:
- Analiza el HOOK (primeros 3 segundos/primera linea)
- Identifica el TRIGGER emocional (curiosidad, FOMO, sorpresa, identificacion)
- Extrae la ESTRUCTURA del contenido
- Nota los CTAs que generan engagement (preguntas, retos, guardados)
- Identifica audios/formatos trending

Responde SOLO en JSON valido con el formato especificado."""

    def _get_content_generation_system_prompt(self, platform: str) -> str:
        platform_specifics = {
            "instagram": """Para Instagram:
- Reels: Hook visual POTENTE en frame 1. Texto grande legible. 15-30 segundos.
- Carousels: Primera slide debe ser irresistible. 5-10 slides. Ultimo slide = CTA.
- Static: Imagen llamativa + caption con valor. Usar emojis estrategicamente.
- Stories: Polls, preguntas, stickers interactivos.""",

            "tiktok": """Para TikTok:
- Videos 15-60 segundos. Hook en segundo 1.
- Trending sounds = mas reach.
- Texto en pantalla durante todo el video.
- CTAs: "Sigueme para parte 2", "Comenta X si..."
- Duet/Stitch = engagement facil.""",

            "linkedin": """Para LinkedIn:
- Posts largos con espaciado visual (saltos de linea).
- Carousels educativos 5-12 slides.
- Hook controversial o insight unico.
- CTA: pregunta para comentarios."""
        }

        return f"""Eres el mejor creador de contenido viral para negocios locales en España y Latinoamerica.

{platform_specifics.get(platform, platform_specifics['instagram'])}

REGLAS CRITICAS:
1. HOOK primero. Si no engancha en 1-3 segundos, nadie ve el resto.
2. Contenido que se puede filmar con MOVIL en menos de 1 hora.
3. CTAs que generan COMENTARIOS y GUARDADOS (mayor peso en algoritmo).
4. Hashtags locales + nicho + trending (max 20 para IG, 5 para TikTok).
5. Texto en espanol natural, no corporativo. Como habla la gente real.

FRAMEWORKS QUE FUNCIONAN:
- HOOK-VALOR-CTA: Engancha, da valor, pide accion
- BEFORE-AFTER: Transformacion visual irresistible
- POV/STORYTIME: Narrativa personal que conecta
- TIPS/HACKS: Valor practico inmediato
- DETRAS DE CAMARAS: Humaniza el negocio

Responde SOLO en JSON valido."""

    def _build_pattern_analysis_prompt(
        self,
        posts: List[Dict[str, Any]],
        business_type: str,
        platform: str
    ) -> str:
        # Prepare posts summary
        posts_summary = []
        for p in posts[:15]:  # Analyze top 15
            posts_summary.append({
                "caption": p.get("caption", "")[:500],
                "type": p.get("type"),
                "likes": p.get("likes", 0),
                "comments": p.get("comments", 0),
                "saves": p.get("saves", 0),
                "views": p.get("video_views", 0) or p.get("plays", 0),
                "audio": p.get("audio_name"),
                "hashtags": p.get("hashtags", [])[:10],
            })

        return f"""Analiza estos {len(posts_summary)} posts TOP de competidores en {platform} para un negocio tipo "{business_type}":

POSTS A ANALIZAR:
{json.dumps(posts_summary, ensure_ascii=False, indent=2)}

Extrae patrones y responde en JSON:
{{
    "hook_patterns": [
        {{
            "pattern_name": "nombre descriptivo",
            "description": "como funciona",
            "examples": ["ejemplo 1", "ejemplo 2"],
            "avg_engagement": numero,
            "best_for": ["awareness", "leads", "engagement"]
        }}
    ],
    "cta_patterns": [
        {{
            "pattern_name": "nombre",
            "description": "descripcion",
            "examples": ["ejemplo"],
            "effectiveness": "alta/media/baja"
        }}
    ],
    "content_pillars": [
        {{
            "pillar_name": "nombre",
            "description": "que tipo de contenido",
            "frequency": "% de posts exitosos",
            "example_topics": ["tema1", "tema2"]
        }}
    ],
    "visual_patterns": [
        {{
            "pattern_name": "nombre",
            "description": "elementos visuales",
            "usage_frequency": "alta/media/baja"
        }}
    ],
    "audio_trends": [
        {{
            "audio_type": "trending/original/voiceover",
            "examples": ["nombre audio"],
            "impact_on_reach": "alto/medio/bajo"
        }}
    ],
    "caption_structure": {{
        "avg_length": "corto/medio/largo",
        "emoji_usage": "alto/medio/bajo",
        "line_breaks": "muchos/pocos",
        "question_usage": "frecuente/ocasional/raro"
    }},
    "best_posting_times": ["lunes 9am", "miercoles 7pm"],
    "hashtag_strategy": {{
        "avg_count": numero,
        "mix": "local + nicho + trending",
        "top_hashtags": ["hashtag1", "hashtag2"]
    }},
    "key_insights": [
        "insight 1 sobre que funciona",
        "insight 2",
        "insight 3"
    ]
}}"""

    def _build_content_generation_prompt(
        self,
        business_info: Dict[str, Any],
        patterns: List[Dict[str, Any]],
        platform: str,
        content_format: str,
        goal: str,
        similar_posts: List[Dict[str, Any]] = None,
        ml_recommendations: Dict[str, Any] = None
    ) -> str:
        # Build ML recommendations section if available
        ml_section = ""
        if ml_recommendations:
            ml_section = f"""
ML RECOMMENDATIONS (MUST INCORPORATE):
- Recommended format: {ml_recommendations.get('recommended_format', content_format)}
- Include these engagement triggers: {', '.join(ml_recommendations.get('trigger_suggestions', []))}
- Optimization tips from ML analysis:
{chr(10).join(f'  • {tip}' for tip in ml_recommendations.get('optimization_tips', [])[:3])}

IMPORTANT: The ML model has analyzed thousands of high-engagement posts.
Incorporate these recommendations to maximize engagement score.
"""

        return f"""Genera contenido de ALTO ENGAGEMENT para:

NEGOCIO:
- Nombre: {business_info.get('name')}
- Tipo: {business_info.get('business_type')}
- Ubicacion: {business_info.get('location', 'España')}
- Voz de marca: {business_info.get('brand_voice', 'friendly_professional')}

REQUISITOS:
- Plataforma: {platform}
- Formato: {content_format}
- Objetivo: {goal}

PATRONES PROBADOS A USAR:
{json.dumps(patterns[:3], ensure_ascii=False, indent=2)}

{f"POSTS SIMILARES EXITOSOS DE REFERENCIA:" if similar_posts else ""}
{json.dumps(similar_posts[:3], ensure_ascii=False, indent=2) if similar_posts else ""}

{ml_section}

Genera contenido en JSON:
{{
    "title": "titulo interno de referencia",
    "hook_text": "texto del hook (primeros 3 seg o primera linea)",
    "caption": "caption completo con emojis y formato",
    "hashtags": ["lista", "de", "hashtags", "optimizados"],
    "video_script": {{
        "hook_0_3s": "que pasa/dice en segundos 0-3",
        "content_3_20s": "contenido principal",
        "cta_20_30s": "llamada a la accion final"
    }},
    "filming_guide": {{
        "setup": "como preparar la escena",
        "equipment": ["movil", "tripode", "luz natural"],
        "angles": ["descripcion de angulos"],
        "text_overlays": [
            {{"text": "texto", "timing": "0-3s", "position": "centro"}}
        ],
        "b_roll": ["ideas de tomas extra"],
        "duration": "25 segundos"
    }},
    "recommended_audio": "nombre de audio trending o 'audio original con voiceover'",
    "audio_alternatives": ["alternativa 1", "alternativa 2"],
    "image_prompt": "prompt para generar imagen con IA si aplica",
    "cta_type": "comment/save/share/dm/follow",
    "emotional_triggers": ["curiosidad", "identificacion"],
    "framework_used": "HOOK-VALOR-CTA",
    "why_it_will_work": "explicacion de por que este contenido generara engagement basado en los patrones"
}}"""

    # ============ MOCK RESPONSES ============

    def _get_mock_pattern_analysis(self, business_type: str, platform: str) -> Dict[str, Any]:
        """Generate realistic mock pattern analysis"""
        return {
            "hook_patterns": [
                {
                    "pattern_name": "Pregunta provocadora",
                    "description": "Empieza con una pregunta que hace pensar al espectador",
                    "examples": [
                        "¿Sabías que el 80% de las flores que compras duran menos de 3 días?",
                        "¿Por qué nadie te cuenta esto sobre los ramos de novia?"
                    ],
                    "avg_engagement": 8500,
                    "best_for": ["awareness", "engagement"]
                },
                {
                    "pattern_name": "POV/Situación identificable",
                    "description": "POV que el espectador puede vivir o ha vivido",
                    "examples": [
                        "POV: Tu novio te regala flores del super",
                        "POV: Pides un ramo para tu madre y esto es lo que recibes"
                    ],
                    "avg_engagement": 12000,
                    "best_for": ["engagement", "awareness"]
                },
                {
                    "pattern_name": "Transformación/Before-After",
                    "description": "Muestra un cambio dramático que sorprende",
                    "examples": [
                        "De esto... a ESTO",
                        "Le dieron estas flores, mira cómo las transformé"
                    ],
                    "avg_engagement": 15000,
                    "best_for": ["awareness", "leads"]
                }
            ],
            "cta_patterns": [
                {
                    "pattern_name": "Pregunta de comentario",
                    "description": "Pide que comenten algo específico",
                    "examples": ["¿Cuál es tu flor favorita? 🌸", "Comenta tu barrio y te digo si hacemos envío"],
                    "effectiveness": "alta"
                },
                {
                    "pattern_name": "Guarda para después",
                    "description": "Pide que guarden el contenido",
                    "examples": ["Guarda este video para tu próximo aniversario 💐", "Guarda si te pasa"],
                    "effectiveness": "alta"
                },
                {
                    "pattern_name": "DM para más info",
                    "description": "Dirige al mensaje directo",
                    "examples": ["DM 'RAMO' para catálogo", "Escríbeme para presupuesto sin compromiso"],
                    "effectiveness": "media"
                }
            ],
            "content_pillars": [
                {
                    "pillar_name": "Tutoriales rápidos",
                    "description": "Tips y trucos en 30 segundos",
                    "frequency": "35%",
                    "example_topics": ["Cómo cuidar tus flores", "DIY arreglos fáciles", "Errores comunes"]
                },
                {
                    "pillar_name": "Behind the scenes",
                    "description": "El día a día del negocio",
                    "frequency": "25%",
                    "example_topics": ["Un día en la floristería", "Preparando pedidos", "Mercado de flores"]
                },
                {
                    "pillar_name": "Transformaciones",
                    "description": "Before/after de arreglos",
                    "frequency": "20%",
                    "example_topics": ["Ramo básico a premium", "Decoración de eventos", "Restauración de flores"]
                },
                {
                    "pillar_name": "Storytelling",
                    "description": "Historias personales y de clientes",
                    "frequency": "20%",
                    "example_topics": ["Mi cliente más especial", "Por qué empecé", "Pedidos memorables"]
                }
            ],
            "visual_patterns": [
                {
                    "pattern_name": "Texto grande en primer frame",
                    "description": "Texto llamativo que se lee en 1 segundo",
                    "usage_frequency": "alta"
                },
                {
                    "pattern_name": "Cara/persona visible",
                    "description": "Mostrar cara humana aumenta conexión",
                    "usage_frequency": "alta"
                },
                {
                    "pattern_name": "Colores saturados",
                    "description": "Flores con colores vivos destacan en feed",
                    "usage_frequency": "alta"
                }
            ],
            "audio_trends": [
                {
                    "audio_type": "trending",
                    "examples": ["Espresso - Sabrina Carpenter", "Flowers - Miley Cyrus"],
                    "impact_on_reach": "alto"
                },
                {
                    "audio_type": "voiceover original",
                    "examples": ["Narración personal explicando el proceso"],
                    "impact_on_reach": "medio"
                }
            ],
            "caption_structure": {
                "avg_length": "medio",
                "emoji_usage": "alto",
                "line_breaks": "muchos",
                "question_usage": "frecuente"
            },
            "best_posting_times": ["martes 11am", "jueves 7pm", "domingo 10am"],
            "hashtag_strategy": {
                "avg_count": 15,
                "mix": "5 locales + 5 nicho + 5 trending",
                "top_hashtags": ["floristeria", "flores", "ramosdenovias", "barcelona", "decoracionfloral"]
            },
            "key_insights": [
                "Los videos con transformaciones tienen 3x más engagement que posts estáticos",
                "Preguntar algo específico en el caption aumenta comentarios un 150%",
                "El contenido behind the scenes humaniza y genera más guardados",
                "Los posts entre 11am-1pm y 7pm-9pm tienen mejor rendimiento",
                "Usar audio trending en las primeras 24h de tendencia multiplica el reach"
            ]
        }

    def _get_mock_generated_content(
        self,
        business_info: Dict[str, Any],
        platform: str,
        content_format: str,
        goal: str
    ) -> Dict[str, Any]:
        """Generate realistic mock content"""
        business_name = business_info.get("name", "Tu Floristería")
        business_type = business_info.get("business_type", "floristeria")

        content_templates = {
            "floristeria": {
                "reel": {
                    "title": "Transformación ramo básico a premium",
                    "hook_text": "POV: Tu cliente dice 'hazlo especial' 🌸",
                    "caption": f"""POV: Tu cliente dice "hazlo especial" y esto es lo que pasa ✨

De un ramo simple a una OBRA DE ARTE en 2 minutos 💐

El secreto está en:
→ Añadir texturas (eucalipto, gypsophila)
→ Variar las alturas
→ Elegir un punto focal

¿Cuál es tu flor favorita para ramos? 👇

Guarda este video para tu próximo regalo 🎁

#{business_name.lower().replace(' ', '')} #floristeria #ramosdenovias #flores #transformacion #barcelona #tendencias2026 #regalosespeciales #decoracionfloral #floresnatirales #fyp""",
                    "hashtags": [
                        business_name.lower().replace(' ', ''),
                        "floristeria", "ramosdenovias", "flores", "transformacion",
                        "barcelona", "tendencias2026", "regalosespeciales",
                        "decoracionfloral", "floresnaturales", "fyp"
                    ],
                    "video_script": {
                        "hook_0_3s": "Mostrar ramo básico con texto 'POV: Tu cliente dice hazlo especial' - expresión de desafío aceptado",
                        "content_3_20s": "Timelapse añadiendo eucalipto, cortando tallos en ángulo, añadiendo gypsophila, ajustando alturas. Texto overlay con cada paso.",
                        "cta_20_30s": "Reveal final del ramo transformado. Texto: '¿Cuál es tu flor favorita?' + manos sosteniendo el ramo"
                    },
                    "filming_guide": {
                        "setup": "Mesa con fondo neutro (blanco o madera clara). Ramo simple ya preparado + flores extra para añadir.",
                        "equipment": ["Móvil con buena cámara", "Trípode o soporte", "Luz natural de ventana o aro de luz"],
                        "angles": [
                            "Cenital (desde arriba) para el trabajo en mesa",
                            "Frontal para el reveal final",
                            "Detalle 45° para close-ups de flores"
                        ],
                        "text_overlays": [
                            {"text": "POV: Tu cliente dice 'hazlo especial'", "timing": "0-3s", "position": "centro"},
                            {"text": "Paso 1: Textura", "timing": "4-8s", "position": "abajo"},
                            {"text": "Paso 2: Alturas", "timing": "9-14s", "position": "abajo"},
                            {"text": "El resultado ✨", "timing": "20-25s", "position": "centro"}
                        ],
                        "b_roll": ["Close-up de manos trabajando", "Detalle de pétalos", "Cliente recibiendo (si posible)"],
                        "duration": "25-30 segundos"
                    },
                    "recommended_audio": "Espresso - Sabrina Carpenter",
                    "audio_alternatives": ["Flowers - Miley Cyrus", "LADY GAGA - Die With A Smile", "Audio original con voiceover"],
                    "image_prompt": "Professional florist hands arranging a beautiful bouquet transformation, before and after split image, soft natural lighting, pastel colors, editorial style photography",
                    "cta_type": "comment",
                    "emotional_triggers": ["curiosidad", "satisfaccion", "aspiracion"],
                    "framework_used": "TRANSFORMATION + HOOK-VALOR-CTA",
                    "why_it_will_work": "Las transformaciones generan el mayor engagement en contenido de floristería porque satisfacen la curiosidad visual. El hook POV crea identificación inmediata. El CTA de pregunta específica ('flor favorita') es fácil de responder y genera comentarios."
                }
            }
        }

        # Default response
        default = content_templates.get("floristeria", {}).get("reel", {})
        return content_templates.get(business_type, content_templates["floristeria"]).get(content_format, default)

    def _get_mock_engagement_prediction(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock engagement prediction"""
        return {
            "score": 87,
            "confidence": 82,
            "breakdown": {
                "hook_score": 92,
                "cta_score": 85,
                "format_score": 88,
                "trend_alignment": 84,
                "emotional_triggers": 89
            },
            "explanation": "Este contenido tiene un 87% de probabilidad de superar los 500 likes porque: 1) Usa el formato POV/transformación que está funcionando muy bien en el nicho de floristería (el video de @competidor_floristeria con este formato consiguió 12k likes), 2) El hook visual de antes/después genera curiosidad inmediata, 3) El CTA de pregunta específica ('flor favorita') es muy fácil de responder, lo que aumentará comentarios.",
            "strengths": [
                "Hook visual potente que genera curiosidad",
                "Formato probado con alto engagement en el nicho",
                "CTA claro que facilita la interacción",
                "Uso de audio trending actual"
            ],
            "improvements": [
                "Considerar añadir texto más grande en el primer frame para mobile",
                "Incluir el precio o rango para filtrar leads calificados si el objetivo es ventas"
            ],
            "predicted_metrics": {
                "likes_range": "800-1500",
                "comments_range": "80-150",
                "saves_estimate": "300-500",
                "shares_estimate": "50-100"
            }
        }

    def _get_mock_viral_ideas(self, keyword: str, business_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate mock viral content ideas"""
        business_type = business_info.get("business_type", "floristeria")

        return [
            {
                "idea_title": f"POV trending adaptado a {business_type}",
                "platform": "tiktok",
                "format": "tiktok_video",
                "hook": f"POV: Trabajas en una {business_type} y un cliente te pide lo imposible",
                "script_outline": [
                    "0-3s: Cara de sorpresa mirando a cámara con texto del hook",
                    "3-10s: Mostrar la situación 'imposible' con humor",
                    "10-20s: Resolver creativamente el problema",
                    "20-25s: Reveal final con texto '¿Lo logré?'"
                ],
                "filming_tips": "Filmar en vertical, expresiones faciales exageradas, cortes rápidos entre escenas",
                "why_it_will_work": f"El formato POV está trending con 2M+ de videos. Adaptarlo a {business_type} es contenido fresco que el algoritmo favorece.",
                "difficulty": "facil",
                "time_to_create": "30 minutos",
                "trending_audio_suggestion": "Oh No - Kreepa",
                "hashtags": [keyword.replace(" ", ""), "pov", "fyp", "viral", business_type, "trabajo", "humor"]
            },
            {
                "idea_title": "5 cosas que no sabías",
                "platform": "instagram",
                "format": "reel",
                "hook": f"5 cosas que NO sabías sobre {keyword} 🤯",
                "script_outline": [
                    "0-3s: Hook con texto grande y expresión de 'te voy a volar la mente'",
                    "3-7s: Dato 1 (el más sorprendente)",
                    "7-12s: Dato 2 con visual",
                    "12-17s: Dato 3",
                    "17-22s: Datos 4 y 5 rápidos",
                    "22-28s: CTA '¿Cuál te sorprendió más?'"
                ],
                "filming_tips": "Usar números grandes en pantalla, transiciones rápidas, mantener energía alta",
                "why_it_will_work": "El formato 'X cosas que no sabías' tiene engagement consistente porque promete valor inmediato y es fácil de consumir.",
                "difficulty": "facil",
                "time_to_create": "45 minutos",
                "trending_audio_suggestion": "Trending instrumental",
                "hashtags": [keyword.replace(" ", ""), "tips", "curiosidades", "aprender", "datos", business_type]
            },
            {
                "idea_title": "Respuesta a comentario viral",
                "platform": "tiktok",
                "format": "tiktok_video",
                "hook": "Alguien me preguntó esto y TENGO que responder...",
                "script_outline": [
                    "0-2s: Mostrar comentario (real o simulado) con cara de 'tengo algo que decir'",
                    "2-15s: Responder con contenido de valor sobre el tema",
                    "15-20s: Plot twist o dato extra que sorprenda",
                    "20-25s: 'Sígueme para más respuestas'"
                ],
                "filming_tips": "Filmar como si hablaras a un amigo, close-up de cara, buena iluminación frontal",
                "why_it_will_work": "El formato de respuesta a comentarios tiene boost algorítmico en TikTok y crea sensación de comunidad.",
                "difficulty": "facil",
                "time_to_create": "20 minutos",
                "trending_audio_suggestion": "audio original",
                "hashtags": [keyword.replace(" ", ""), "respuesta", "fyp", "preguntas", business_type, "aprender"]
            }
        ]
