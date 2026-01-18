#!/usr/bin/env python3
"""
=============================================================================
APIFY INSTAGRAM DISCOVERY - Descubrimiento de Perfiles Locales
=============================================================================

Script para descubrir perfiles de Instagram relevantes en un nicho de negocio
local (inmobiliarias, floristerías, restaurantes, gimnasios, etc.) usando
EXCLUSIVAMENTE el Apify Client.

OBJETIVO: Generar lista de usernames de competidores locales similares
(cuentas pequeñas/medianas, no influencers grandes) para análisis manual.

FLUJO:
1. Input: hashtags relevantes + keywords de ubicación/nicho
2. Scraping de posts recientes por hashtag (límite bajo: 100-200 posts)
3. Extracción de usernames únicos
4. (Opcional) Enriquecimiento con datos de perfil
5. Filtrado por criterios locales (followers, bio, categoría)
6. Output: CSV/JSON con perfiles relevantes

IMPORTANTE: Este script NO extrae contenido de posts/stories.
Solo sirve para DESCUBRIR perfiles que la usuaria visitará manualmente.

Uso:
    export APIFY_API_TOKEN="tu_token_aquí"
    python apify_instagram_discovery.py

Autor: Elena Redes / BrandPulse AI
Fecha: 2026-01
=============================================================================
"""

import os
import sys
import json
import csv
import time
import logging
import re
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

@dataclass
class DiscoveryConfig:
    """
    Configuración del proceso de discovery.
    Personaliza estos valores según tu nicho y ubicación.
    """
    # === HASHTAGS A BUSCAR ===
    # Hashtags relevantes para tu nicho (sin #)
    hashtags: List[str] = field(default_factory=lambda: [
        # Ejemplos para Almería - PERSONALIZA SEGÚN TU NICHO
        "inmobiliariaalmeria",
        "casasalmeria",
        "pisosalmeria",
        "alquileralmeria",
        "ventacasasalmeria",
        # Añade más según necesites
    ])

    # === KEYWORDS DE UBICACIÓN (para filtrar bio) ===
    location_keywords: List[str] = field(default_factory=lambda: [
        "almería", "almeria", "andalucía", "andalucia",
        "roquetas", "aguadulce", "níjar", "nijar",
        "ejido", "mojacar", "cabo de gata",
        # Añade pueblos/zonas de tu área
    ])

    # === KEYWORDS DE NICHO (para filtrar bio) ===
    niche_keywords: List[str] = field(default_factory=lambda: [
        # Ejemplo para inmobiliarias - CAMBIA SEGÚN TU NICHO
        "inmobiliaria", "inmuebles", "casas", "pisos",
        "alquiler", "venta", "propiedades", "real estate",
        "vivienda", "hogar", "apartamentos",
        # Para otros nichos:
        # Floristerías: "floristería", "flores", "ramos", "bodas"
        # Restaurantes: "restaurante", "gastronomía", "cocina"
        # Gimnasios: "gimnasio", "fitness", "entrenamiento"
    ])

    # === LÍMITES DE SCRAPING ===
    max_posts_per_hashtag: int = 150  # Posts a extraer por hashtag (100-200 recomendado)
    max_profiles_output: int = 150    # Máximo perfiles en output final

    # === FILTROS DE TAMAÑO DE CUENTA ===
    min_followers: int = 100          # Mínimo followers (evitar cuentas vacías)
    max_followers: int = 50000        # Máximo followers (evitar influencers)

    # === ENRIQUECIMIENTO DE PERFILES ===
    enrich_profiles: bool = True      # Obtener datos adicionales de cada perfil
    enrich_batch_size: int = 10       # Perfiles a enriquecer por batch

    # === RATE LIMITING ===
    delay_between_requests: float = 2.0   # Segundos entre requests
    max_retries: int = 3                   # Reintentos por request
    retry_delay: float = 5.0               # Segundos entre reintentos

    # === OUTPUT ===
    output_dir: str = "./discovery_output"
    output_prefix: str = "instagram_discovery"


@dataclass
class DiscoveredProfile:
    """
    Perfil descubierto durante el proceso.
    """
    username: str
    full_name: str = ""
    followers: int = 0
    following: int = 0
    posts_count: int = 0
    bio: str = ""
    category: str = ""
    profile_url: str = ""
    external_url: str = ""
    is_business: bool = False
    is_verified: bool = False
    location_detected: str = ""
    niche_detected: str = ""
    relevance_score: float = 0.0
    discovered_via_hashtag: str = ""
    discovered_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# CLIENTE APIFY
# =============================================================================

class ApifyDiscoveryClient:
    """
    Cliente para interactuar con Apify API para discovery de perfiles.
    """

    # Actores de Apify para Instagram
    HASHTAG_SCRAPER = "apify/instagram-hashtag-scraper"
    PROFILE_SCRAPER = "apify/instagram-profile-scraper"

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("APIFY_API_TOKEN")

        if not self.api_token:
            raise ValueError(
                "APIFY_API_TOKEN no configurado. "
                "Configúralo como variable de entorno o pásalo al constructor."
            )

        try:
            from apify_client import ApifyClient
            self.client = ApifyClient(self.api_token)
            logger.info("Cliente Apify inicializado correctamente")
        except ImportError:
            raise ImportError(
                "apify-client no está instalado. "
                "Instálalo con: pip install apify-client"
            )

    def scrape_hashtag_posts(
        self,
        hashtag: str,
        max_posts: int = 100,
        retries: int = 3,
        retry_delay: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Extrae posts recientes de un hashtag específico.

        Args:
            hashtag: Hashtag a buscar (sin #)
            max_posts: Número máximo de posts a extraer
            retries: Número de reintentos en caso de error
            retry_delay: Segundos entre reintentos

        Returns:
            Lista de posts con información del owner
        """
        logger.info(f"Buscando posts en #{hashtag} (máx: {max_posts})")

        run_input = {
            "hashtags": [hashtag],
            "resultsLimit": max_posts,
            "resultsType": "posts",
            # Solo necesitamos info básica del post y owner
            "extendOutputFunction": """
                async ({ data, item, page, request, customData }) => {
                    return item;
                }
            """,
        }

        for attempt in range(1, retries + 1):
            try:
                # Ejecutar el actor
                run = self.client.actor(self.HASHTAG_SCRAPER).call(
                    run_input=run_input,
                    timeout_secs=300,  # 5 minutos máximo
                )

                # Obtener resultados
                items = list(
                    self.client.dataset(run["defaultDatasetId"]).iterate_items()
                )

                logger.info(f"#{hashtag}: Obtenidos {len(items)} posts")
                return items

            except Exception as e:
                logger.warning(
                    f"#{hashtag}: Error en intento {attempt}/{retries}: {e}"
                )
                if attempt < retries:
                    logger.info(f"Reintentando en {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"#{hashtag}: Fallaron todos los intentos")
                    return []

        return []

    def scrape_profile_details(
        self,
        usernames: List[str],
        retries: int = 3,
        retry_delay: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Obtiene detalles de perfiles específicos.

        Args:
            usernames: Lista de usernames a consultar
            retries: Número de reintentos
            retry_delay: Segundos entre reintentos

        Returns:
            Lista de perfiles con información detallada
        """
        if not usernames:
            return []

        logger.info(f"Obteniendo detalles de {len(usernames)} perfiles")

        # Construir URLs de perfiles
        profile_urls = [
            f"https://www.instagram.com/{username}/"
            for username in usernames
        ]

        run_input = {
            "directUrls": profile_urls,
            "resultsType": "details",
            "resultsLimit": len(usernames),
        }

        for attempt in range(1, retries + 1):
            try:
                run = self.client.actor(self.PROFILE_SCRAPER).call(
                    run_input=run_input,
                    timeout_secs=300,
                )

                items = list(
                    self.client.dataset(run["defaultDatasetId"]).iterate_items()
                )

                logger.info(f"Detalles obtenidos: {len(items)} perfiles")
                return items

            except Exception as e:
                logger.warning(
                    f"Profile details: Error en intento {attempt}/{retries}: {e}"
                )
                if attempt < retries:
                    time.sleep(retry_delay)
                else:
                    logger.error("Profile details: Fallaron todos los intentos")
                    return []

        return []


# =============================================================================
# MOTOR DE DISCOVERY
# =============================================================================

class InstagramDiscoveryEngine:
    """
    Motor principal para descubrir perfiles relevantes en Instagram.
    """

    def __init__(self, config: DiscoveryConfig):
        self.config = config
        self.client = ApifyDiscoveryClient()
        self.discovered_usernames: Set[str] = set()
        self.profiles: Dict[str, DiscoveredProfile] = {}

        # Crear directorio de output
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def run_discovery(self) -> List[DiscoveredProfile]:
        """
        Ejecuta el proceso completo de discovery.

        Returns:
            Lista de perfiles descubiertos y filtrados
        """
        logger.info("=" * 60)
        logger.info("INICIANDO DISCOVERY DE PERFILES INSTAGRAM")
        logger.info("=" * 60)
        logger.info(f"Hashtags a buscar: {len(self.config.hashtags)}")
        logger.info(f"Max posts/hashtag: {self.config.max_posts_per_hashtag}")
        logger.info(f"Filtro followers: {self.config.min_followers}-{self.config.max_followers}")
        logger.info("=" * 60)

        # Paso 1: Extraer posts de todos los hashtags
        all_posts = self._scrape_all_hashtags()

        if not all_posts:
            logger.warning("No se obtuvieron posts. Verifica los hashtags y tu API token.")
            return []

        # Paso 2: Extraer usernames únicos
        self._extract_unique_usernames(all_posts)

        logger.info(f"Usernames únicos encontrados: {len(self.discovered_usernames)}")

        if not self.discovered_usernames:
            logger.warning("No se encontraron usernames en los posts.")
            return []

        # Paso 3: Enriquecer perfiles (opcional)
        if self.config.enrich_profiles:
            self._enrich_profiles()

        # Paso 4: Calcular relevancia y filtrar
        relevant_profiles = self._filter_and_score_profiles()

        # Paso 5: Guardar resultados
        self._save_results(relevant_profiles)

        logger.info("=" * 60)
        logger.info(f"DISCOVERY COMPLETADO: {len(relevant_profiles)} perfiles relevantes")
        logger.info("=" * 60)

        return relevant_profiles

    def _scrape_all_hashtags(self) -> List[Dict[str, Any]]:
        """
        Extrae posts de todos los hashtags configurados.
        """
        all_posts = []

        for i, hashtag in enumerate(self.config.hashtags, 1):
            logger.info(f"[{i}/{len(self.config.hashtags)}] Procesando #{hashtag}")

            posts = self.client.scrape_hashtag_posts(
                hashtag=hashtag,
                max_posts=self.config.max_posts_per_hashtag,
                retries=self.config.max_retries,
                retry_delay=self.config.retry_delay,
            )

            # Marcar de qué hashtag viene cada post
            for post in posts:
                post["_discovered_via_hashtag"] = hashtag

            all_posts.extend(posts)

            # Rate limiting entre hashtags
            if i < len(self.config.hashtags):
                logger.info(f"Esperando {self.config.delay_between_requests}s...")
                time.sleep(self.config.delay_between_requests)

        logger.info(f"Total posts obtenidos: {len(all_posts)}")
        return all_posts

    def _extract_unique_usernames(self, posts: List[Dict[str, Any]]) -> None:
        """
        Extrae usernames únicos de los posts y crea perfiles básicos.
        """
        for post in posts:
            # El username puede estar en diferentes campos según el actor
            username = (
                post.get("ownerUsername") or
                post.get("username") or
                post.get("owner", {}).get("username")
            )

            if not username:
                continue

            # Normalizar username
            username = username.lower().strip()

            # Evitar duplicados
            if username in self.discovered_usernames:
                continue

            self.discovered_usernames.add(username)

            # Crear perfil básico con datos disponibles del post
            profile = DiscoveredProfile(
                username=username,
                full_name=post.get("ownerFullName", ""),
                followers=post.get("ownerFollowerCount", 0),
                profile_url=f"https://www.instagram.com/{username}/",
                is_verified=post.get("ownerIsVerified", False),
                discovered_via_hashtag=post.get("_discovered_via_hashtag", ""),
                discovered_at=datetime.now().isoformat(),
            )

            self.profiles[username] = profile

    def _enrich_profiles(self) -> None:
        """
        Enriquece los perfiles con datos adicionales del profile scraper.
        """
        usernames = list(self.profiles.keys())
        total_batches = (len(usernames) + self.config.enrich_batch_size - 1) // self.config.enrich_batch_size

        logger.info(f"Enriqueciendo {len(usernames)} perfiles en {total_batches} batches")

        for batch_num in range(total_batches):
            start_idx = batch_num * self.config.enrich_batch_size
            end_idx = min(start_idx + self.config.enrich_batch_size, len(usernames))
            batch_usernames = usernames[start_idx:end_idx]

            logger.info(f"Batch {batch_num + 1}/{total_batches}: {len(batch_usernames)} perfiles")

            try:
                profile_details = self.client.scrape_profile_details(
                    usernames=batch_usernames,
                    retries=self.config.max_retries,
                    retry_delay=self.config.retry_delay,
                )

                # Actualizar perfiles con datos enriquecidos
                for detail in profile_details:
                    username = detail.get("username", "").lower().strip()
                    if username in self.profiles:
                        self._update_profile_from_details(username, detail)

            except Exception as e:
                logger.warning(f"Error enriqueciendo batch {batch_num + 1}: {e}")

            # Rate limiting entre batches
            if batch_num < total_batches - 1:
                time.sleep(self.config.delay_between_requests)

    def _update_profile_from_details(
        self,
        username: str,
        details: Dict[str, Any]
    ) -> None:
        """
        Actualiza un perfil con datos del profile scraper.
        """
        profile = self.profiles[username]

        profile.full_name = details.get("fullName", profile.full_name)
        profile.followers = details.get("followersCount", profile.followers)
        profile.following = details.get("followingCount", 0)
        profile.posts_count = details.get("postsCount", 0)
        profile.bio = details.get("biography", "")
        profile.category = details.get("businessCategoryName", "")
        profile.external_url = details.get("externalUrl", "")
        profile.is_business = details.get("isBusinessAccount", False)
        profile.is_verified = details.get("verified", profile.is_verified)

    def _filter_and_score_profiles(self) -> List[DiscoveredProfile]:
        """
        Filtra perfiles por criterios y calcula puntuación de relevancia.
        """
        logger.info("Filtrando y puntuando perfiles...")

        relevant_profiles = []

        for username, profile in self.profiles.items():
            # === FILTRO 1: Rango de followers ===
            if profile.followers < self.config.min_followers:
                logger.debug(f"@{username}: Descartado (muy pocos followers: {profile.followers})")
                continue

            if profile.followers > self.config.max_followers:
                logger.debug(f"@{username}: Descartado (demasiados followers: {profile.followers})")
                continue

            # === CALCULAR RELEVANCIA ===
            relevance_score, location_detected, niche_detected = self._calculate_relevance(profile)

            profile.relevance_score = relevance_score
            profile.location_detected = location_detected
            profile.niche_detected = niche_detected

            # === FILTRO 2: Relevancia mínima ===
            # Requerimos al menos ubicación O nicho detectado
            if not location_detected and not niche_detected:
                logger.debug(f"@{username}: Descartado (sin indicadores de relevancia)")
                continue

            relevant_profiles.append(profile)

        # Ordenar por relevancia descendente
        relevant_profiles.sort(key=lambda p: p.relevance_score, reverse=True)

        # Limitar cantidad de output
        if len(relevant_profiles) > self.config.max_profiles_output:
            relevant_profiles = relevant_profiles[:self.config.max_profiles_output]
            logger.info(f"Limitado a {self.config.max_profiles_output} perfiles top")

        logger.info(f"Perfiles relevantes después de filtrado: {len(relevant_profiles)}")

        return relevant_profiles

    def _calculate_relevance(
        self,
        profile: DiscoveredProfile
    ) -> tuple[float, str, str]:
        """
        Calcula puntuación de relevancia basada en bio, categoría, etc.

        Returns:
            (score, location_detected, niche_detected)
        """
        score = 0.0
        location_detected = ""
        niche_detected = ""

        # Texto a analizar (bio + nombre + categoría)
        text_to_analyze = " ".join([
            profile.bio.lower(),
            profile.full_name.lower(),
            profile.category.lower(),
            profile.username.lower(),
        ])

        # === PUNTOS POR UBICACIÓN ===
        for keyword in self.config.location_keywords:
            if keyword.lower() in text_to_analyze:
                score += 30  # Ubicación vale mucho
                if not location_detected:
                    location_detected = keyword

        # === PUNTOS POR NICHO ===
        niche_matches = []
        for keyword in self.config.niche_keywords:
            if keyword.lower() in text_to_analyze:
                score += 20
                niche_matches.append(keyword)

        if niche_matches:
            niche_detected = ", ".join(niche_matches[:3])  # Max 3 keywords

        # === BONUS POR CUENTA DE NEGOCIO ===
        if profile.is_business:
            score += 15

        # === BONUS POR CATEGORÍA RELEVANTE ===
        if profile.category:
            score += 10

        # === PENALIZACIÓN POR CUENTA VERIFICADA (probablemente grande) ===
        if profile.is_verified:
            score -= 20

        # === BONUS POR RANGO ÓPTIMO DE FOLLOWERS ===
        # Preferimos cuentas entre 1k-15k (competidores reales, no micro)
        if 1000 <= profile.followers <= 15000:
            score += 10

        # Normalizar score a 0-100
        score = min(max(score, 0), 100)

        return score, location_detected, niche_detected

    def _save_results(self, profiles: List[DiscoveredProfile]) -> None:
        """
        Guarda resultados en CSV y JSON.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.config.output_prefix}_{timestamp}"

        # === GUARDAR CSV ===
        csv_path = Path(self.config.output_dir) / f"{base_name}.csv"

        if profiles:
            fieldnames = [
                "username", "full_name", "followers", "following", "posts_count",
                "bio", "category", "profile_url", "external_url",
                "is_business", "is_verified", "location_detected", "niche_detected",
                "relevance_score", "discovered_via_hashtag", "discovered_at"
            ]

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for profile in profiles:
                    writer.writerow(profile.to_dict())

            logger.info(f"CSV guardado: {csv_path}")

        # === GUARDAR JSON ===
        json_path = Path(self.config.output_dir) / f"{base_name}.json"

        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_profiles": len(profiles),
                "config": {
                    "hashtags": self.config.hashtags,
                    "location_keywords": self.config.location_keywords,
                    "niche_keywords": self.config.niche_keywords,
                    "followers_range": f"{self.config.min_followers}-{self.config.max_followers}",
                },
            },
            "profiles": [p.to_dict() for p in profiles],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON guardado: {json_path}")

        # === RESUMEN EN CONSOLA ===
        print("\n" + "=" * 60)
        print("RESUMEN DE DISCOVERY")
        print("=" * 60)
        print(f"Total perfiles encontrados: {len(profiles)}")
        print(f"Archivos guardados en: {self.config.output_dir}/")
        print(f"  - {csv_path.name}")
        print(f"  - {json_path.name}")
        print("=" * 60)

        if profiles:
            print("\nTOP 10 PERFILES MÁS RELEVANTES:")
            print("-" * 60)
            for i, p in enumerate(profiles[:10], 1):
                print(f"{i:2}. @{p.username:<25} | {p.followers:>6} seg | Score: {p.relevance_score:.0f}")
                if p.location_detected:
                    print(f"    📍 {p.location_detected}")
                if p.niche_detected:
                    print(f"    🏷️  {p.niche_detected}")
            print("-" * 60)


# =============================================================================
# CONFIGURACIONES PREDEFINIDAS POR NICHO
# =============================================================================

def get_config_inmobiliaria_almeria() -> DiscoveryConfig:
    """Configuración para inmobiliarias en Almería."""
    return DiscoveryConfig(
        hashtags=[
            "inmobiliariaalmeria",
            "casasalmeria",
            "pisosalmeria",
            "alquileralmeria",
            "ventaalmeria",
            "propiedadesalmeria",
            "inmobiliariaandalucia",
            "almeriavivienda",
            "apartamentosalmeria",
            "hogaralmeria",
        ],
        location_keywords=[
            "almería", "almeria", "andalucía", "andalucia",
            "roquetas de mar", "roquetas", "aguadulce",
            "ejido", "el ejido", "níjar", "nijar", "mojácar", "mojacar",
            "cabo de gata", "garrucha", "vera", "carboneras",
            "huércal", "huercal", "adra", "vícar", "vicar",
        ],
        niche_keywords=[
            "inmobiliaria", "inmuebles", "real estate", "casas", "pisos",
            "alquiler", "venta", "propiedades", "vivienda", "hogar",
            "apartamentos", "chalets", "áticos", "locales", "oficinas",
            "terrenos", "fincas", "promoción", "obra nueva",
            "inversión inmobiliaria", "gestor", "api", "agente inmobiliario",
        ],
        max_posts_per_hashtag=150,
        max_profiles_output=150,
        min_followers=100,
        max_followers=50000,
    )


def get_config_floristeria_almeria() -> DiscoveryConfig:
    """Configuración para floristerías en Almería."""
    return DiscoveryConfig(
        hashtags=[
            "floristeriaalmeria",
            "floresalmeria",
            "ramosalmeria",
            "bodaalmeria",
            "eventosalmeria",
            "floristeriasandalucia",
        ],
        location_keywords=[
            "almería", "almeria", "andalucía", "andalucia",
            "roquetas", "aguadulce", "ejido", "níjar",
        ],
        niche_keywords=[
            "floristería", "floristeria", "flores", "ramos",
            "bodas", "eventos", "decoración floral", "plantas",
            "arreglos", "coronas", "centros", "envío flores",
        ],
        max_posts_per_hashtag=100,
        max_profiles_output=100,
    )


def get_config_restaurante_almeria() -> DiscoveryConfig:
    """Configuración para restaurantes en Almería."""
    return DiscoveryConfig(
        hashtags=[
            "restaurantealmeria",
            "tapasalmeria",
            "gastronomiaalmeria",
            "comeralmeria",
            "almeriagastronomica",
            "chiringuitoroqueats",
        ],
        location_keywords=[
            "almería", "almeria", "andalucía", "andalucia",
            "roquetas", "aguadulce", "cabo de gata", "mojácar",
        ],
        niche_keywords=[
            "restaurante", "tapas", "gastronomía", "cocina",
            "chef", "comida", "menú", "terraza", "bar",
            "mariscos", "pescado", "chiringuito",
        ],
        max_posts_per_hashtag=120,
        max_profiles_output=100,
    )


def get_config_gimnasio_almeria() -> DiscoveryConfig:
    """Configuración para gimnasios en Almería."""
    return DiscoveryConfig(
        hashtags=[
            "gimnasioalmeria",
            "fitnessalmeria",
            "crossfitalmeria",
            "entrenamientoalmeria",
            "personaltraineralmeria",
        ],
        location_keywords=[
            "almería", "almeria", "andalucía", "andalucia",
            "roquetas", "aguadulce", "ejido",
        ],
        niche_keywords=[
            "gimnasio", "gym", "fitness", "crossfit",
            "entrenamiento", "personal trainer", "musculación",
            "cardio", "spinning", "pilates", "yoga",
        ],
        max_posts_per_hashtag=100,
        max_profiles_output=80,
    )


# =============================================================================
# MAIN - PUNTO DE ENTRADA
# =============================================================================

def main():
    """
    Punto de entrada principal del script.
    """
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     APIFY INSTAGRAM DISCOVERY - Perfiles Locales          ║
    ║     Elena Redes / BrandPulse AI                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Verificar token de API
    if not os.environ.get("APIFY_API_TOKEN"):
        print("⚠️  ERROR: APIFY_API_TOKEN no está configurado.")
        print("   Configúralo con: export APIFY_API_TOKEN='tu_token_aquí'")
        sys.exit(1)

    # =========================================================================
    # SELECCIONA TU CONFIGURACIÓN AQUÍ
    # =========================================================================
    # Descomenta la línea correspondiente a tu nicho:

    config = get_config_inmobiliaria_almeria()    # 🏠 Inmobiliarias
    # config = get_config_floristeria_almeria()   # 🌸 Floristerías
    # config = get_config_restaurante_almeria()   # 🍽️ Restaurantes
    # config = get_config_gimnasio_almeria()      # 💪 Gimnasios

    # O crea tu propia configuración personalizada:
    # config = DiscoveryConfig(
    #     hashtags=["tuhashtag1", "tuhashtag2"],
    #     location_keywords=["almería", "tu_zona"],
    #     niche_keywords=["tu_nicho", "keywords"],
    # )

    # =========================================================================

    # Crear motor y ejecutar discovery
    try:
        engine = InstagramDiscoveryEngine(config)
        profiles = engine.run_discovery()

        if profiles:
            print(f"\n✅ Discovery completado exitosamente!")
            print(f"   {len(profiles)} perfiles relevantes encontrados.")
            print(f"   Revisa los archivos en: {config.output_dir}/")
        else:
            print("\n⚠️  No se encontraron perfiles relevantes.")
            print("   Prueba con otros hashtags o ajusta los filtros.")

    except ValueError as e:
        print(f"\n❌ Error de configuración: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\n❌ Dependencia faltante: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Error inesperado durante el discovery")
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
