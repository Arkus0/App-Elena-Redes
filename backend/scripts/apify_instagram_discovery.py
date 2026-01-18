#!/usr/bin/env python3
"""
=============================================================================
APIFY INSTAGRAM DISCOVERY - Descubrimiento de Perfiles Locales
=============================================================================

Script para descubrir perfiles de Instagram relevantes en un nicho de negocio
local usando EXCLUSIVAMENTE el Apify Client.

OBJETIVO: Generar lista de usernames de competidores locales similares
(cuentas pequeñas/medianas, no influencers grandes) para análisis manual
posterior con la extensión Elena Bridge.

MODOS DE USO:
    1. Interactivo (pregunta qué buscar):
       python apify_instagram_discovery.py

    2. Con archivo de configuración:
       python apify_instagram_discovery.py --config mi_busqueda.yaml

    3. Con argumentos CLI:
       python apify_instagram_discovery.py --hashtags "inmobiliariaalmeria,casasalmeria" \
           --location "almería,roquetas" --niche "inmobiliaria,casas"

IMPORTANTE: Este script solo DESCUBRE perfiles. La extracción de contenido
individual se hace con la extensión Elena Bridge Chrome.

=============================================================================
"""

import os
import sys
import json
import csv
import time
import logging
import argparse
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
    Puede cargarse desde YAML, JSON, CLI o modo interactivo.
    """
    # === HASHTAGS A BUSCAR ===
    hashtags: List[str] = field(default_factory=list)

    # === KEYWORDS DE UBICACIÓN (para filtrar bio) ===
    location_keywords: List[str] = field(default_factory=list)

    # === KEYWORDS DE NICHO (para filtrar bio) ===
    niche_keywords: List[str] = field(default_factory=list)

    # === LÍMITES DE SCRAPING ===
    max_posts_per_hashtag: int = 150
    max_profiles_output: int = 150

    # === FILTROS DE TAMAÑO DE CUENTA ===
    min_followers: int = 100
    max_followers: int = 50000

    # === ENRIQUECIMIENTO DE PERFILES ===
    enrich_profiles: bool = True
    enrich_batch_size: int = 10

    # === RATE LIMITING ===
    delay_between_requests: float = 2.0
    max_retries: int = 3
    retry_delay: float = 5.0

    # === OUTPUT ===
    output_dir: str = "./discovery_output"
    output_prefix: str = "instagram_discovery"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_to_yaml(self, path: str) -> None:
        """Guarda la configuración a un archivo YAML."""
        try:
            import yaml
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)
            logger.info(f"Configuración guardada en: {path}")
        except ImportError:
            # Fallback a JSON si no hay PyYAML
            json_path = path.replace('.yaml', '.json').replace('.yml', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Configuración guardada en: {json_path}")

    @classmethod
    def from_yaml(cls, path: str) -> 'DiscoveryConfig':
        """Carga configuración desde archivo YAML o JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.json'):
                data = json.load(f)
            else:
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError(
                        "PyYAML no está instalado. Instálalo con: pip install pyyaml\n"
                        "O usa un archivo .json en su lugar."
                    )
        return cls(**data)


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
# MODO INTERACTIVO
# =============================================================================

def run_interactive_setup() -> DiscoveryConfig:
    """
    Modo interactivo: pregunta al usuario qué quiere buscar.
    """
    print("\n" + "=" * 60)
    print("CONFIGURACIÓN INTERACTIVA")
    print("=" * 60)
    print("Responde las siguientes preguntas para configurar la búsqueda.")
    print("Puedes dejar en blanco para usar valores por defecto.\n")

    # === HASHTAGS ===
    print("-" * 40)
    print("1. HASHTAGS A BUSCAR")
    print("-" * 40)
    print("Introduce los hashtags que quieres buscar (sin #).")
    print("Ejemplos: inmobiliariaalmeria, casasalmeria, pisosalmeria")
    print("Separa múltiples hashtags con comas.\n")

    hashtags_input = input("Hashtags: ").strip()
    hashtags = [h.strip().lower().replace("#", "") for h in hashtags_input.split(",") if h.strip()]

    if not hashtags:
        print("No has introducido hashtags. Usando ejemplos por defecto.")
        hashtags = ["inmobiliariaalmeria", "casasalmeria"]

    # === UBICACIÓN ===
    print("\n" + "-" * 40)
    print("2. PALABRAS CLAVE DE UBICACIÓN")
    print("-" * 40)
    print("Palabras que deben aparecer en la bio para considerar el perfil local.")
    print("Ejemplos: almería, roquetas, aguadulce, andalucía")
    print("Separa múltiples palabras con comas.\n")

    location_input = input("Ubicación: ").strip()
    location_keywords = [l.strip().lower() for l in location_input.split(",") if l.strip()]

    if not location_keywords:
        print("No has introducido ubicación. Usando 'almería' por defecto.")
        location_keywords = ["almería", "almeria"]

    # === NICHO ===
    print("\n" + "-" * 40)
    print("3. PALABRAS CLAVE DEL NICHO")
    print("-" * 40)
    print("Palabras que identifican el tipo de negocio.")
    print("Ejemplos para inmobiliarias: inmobiliaria, casas, pisos, alquiler, venta")
    print("Ejemplos para floristerías: floristería, flores, ramos, bodas")
    print("Separa múltiples palabras con comas.\n")

    niche_input = input("Nicho: ").strip()
    niche_keywords = [n.strip().lower() for n in niche_input.split(",") if n.strip()]

    if not niche_keywords:
        print("No has introducido nicho. Usando 'inmobiliaria, casas' por defecto.")
        niche_keywords = ["inmobiliaria", "casas", "pisos"]

    # === FILTROS DE FOLLOWERS ===
    print("\n" + "-" * 40)
    print("4. RANGO DE SEGUIDORES")
    print("-" * 40)
    print("Define el rango de seguidores para filtrar.")
    print("Por defecto: mínimo 100, máximo 50000\n")

    try:
        min_input = input("Mínimo seguidores [100]: ").strip()
        min_followers = int(min_input) if min_input else 100
    except ValueError:
        min_followers = 100

    try:
        max_input = input("Máximo seguidores [50000]: ").strip()
        max_followers = int(max_input) if max_input else 50000
    except ValueError:
        max_followers = 50000

    # === LÍMITES ===
    print("\n" + "-" * 40)
    print("5. LÍMITES DE BÚSQUEDA")
    print("-" * 40)

    try:
        posts_input = input("Posts por hashtag [150]: ").strip()
        max_posts = int(posts_input) if posts_input else 150
    except ValueError:
        max_posts = 150

    try:
        profiles_input = input("Máximo perfiles en output [150]: ").strip()
        max_profiles = int(profiles_input) if profiles_input else 150
    except ValueError:
        max_profiles = 150

    # === ENRIQUECIMIENTO ===
    print("\n" + "-" * 40)
    print("6. ENRIQUECIMIENTO DE PERFILES")
    print("-" * 40)
    print("¿Quieres obtener datos adicionales de cada perfil (bio, followers exactos)?")
    print("Esto consume más créditos de Apify pero da mejores resultados.\n")

    enrich_input = input("Enriquecer perfiles? [S/n]: ").strip().lower()
    enrich_profiles = enrich_input not in ['n', 'no']

    # === CREAR CONFIG ===
    config = DiscoveryConfig(
        hashtags=hashtags,
        location_keywords=location_keywords,
        niche_keywords=niche_keywords,
        min_followers=min_followers,
        max_followers=max_followers,
        max_posts_per_hashtag=max_posts,
        max_profiles_output=max_profiles,
        enrich_profiles=enrich_profiles,
    )

    # === GUARDAR CONFIG ===
    print("\n" + "-" * 40)
    print("7. GUARDAR CONFIGURACIÓN")
    print("-" * 40)
    print("¿Quieres guardar esta configuración para reutilizarla?\n")

    save_input = input("Guardar config? [s/N]: ").strip().lower()
    if save_input in ['s', 'si', 'sí', 'yes', 'y']:
        config_name = input("Nombre del archivo [mi_busqueda.yaml]: ").strip()
        if not config_name:
            config_name = "mi_busqueda.yaml"
        if not config_name.endswith(('.yaml', '.yml', '.json')):
            config_name += ".yaml"

        Path("./discovery_configs").mkdir(exist_ok=True)
        config_path = f"./discovery_configs/{config_name}"
        config.save_to_yaml(config_path)

    # === RESUMEN ===
    print("\n" + "=" * 60)
    print("RESUMEN DE CONFIGURACIÓN")
    print("=" * 60)
    print(f"Hashtags: {', '.join(hashtags)}")
    print(f"Ubicación: {', '.join(location_keywords)}")
    print(f"Nicho: {', '.join(niche_keywords)}")
    print(f"Seguidores: {min_followers} - {max_followers}")
    print(f"Posts/hashtag: {max_posts}")
    print(f"Max perfiles: {max_profiles}")
    print(f"Enriquecer: {'Sí' if enrich_profiles else 'No'}")
    print("=" * 60)

    confirm = input("\n¿Continuar con esta configuración? [S/n]: ").strip().lower()
    if confirm in ['n', 'no']:
        print("Búsqueda cancelada.")
        sys.exit(0)

    return config


# =============================================================================
# CLIENTE APIFY
# =============================================================================

class ApifyDiscoveryClient:
    """
    Cliente para interactuar con Apify API para discovery de perfiles.
    """

    HASHTAG_SCRAPER = "apify/instagram-hashtag-scraper"
    PROFILE_SCRAPER = "apify/instagram-profile-scraper"

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("APIFY_API_TOKEN")

        if not self.api_token:
            raise ValueError(
                "APIFY_API_TOKEN no configurado. "
                "Configúralo con: export APIFY_API_TOKEN='tu_token_aquí'"
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
        """Extrae posts recientes de un hashtag específico."""
        logger.info(f"Buscando posts en #{hashtag} (máx: {max_posts})")

        run_input = {
            "hashtags": [hashtag],
            "resultsLimit": max_posts,
            "resultsType": "posts",
        }

        for attempt in range(1, retries + 1):
            try:
                run = self.client.actor(self.HASHTAG_SCRAPER).call(
                    run_input=run_input,
                    timeout_secs=300,
                )

                items = list(
                    self.client.dataset(run["defaultDatasetId"]).iterate_items()
                )

                logger.info(f"#{hashtag}: Obtenidos {len(items)} posts")
                return items

            except Exception as e:
                logger.warning(f"#{hashtag}: Error en intento {attempt}/{retries}: {e}")
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
        """Obtiene detalles de perfiles específicos."""
        if not usernames:
            return []

        logger.info(f"Obteniendo detalles de {len(usernames)} perfiles")

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
                logger.warning(f"Profile details: Error en intento {attempt}/{retries}: {e}")
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
    """Motor principal para descubrir perfiles relevantes en Instagram."""

    def __init__(self, config: DiscoveryConfig):
        self.config = config
        self.client = ApifyDiscoveryClient()
        self.discovered_usernames: Set[str] = set()
        self.profiles: Dict[str, DiscoveredProfile] = {}

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def run_discovery(self) -> List[DiscoveredProfile]:
        """Ejecuta el proceso completo de discovery."""
        logger.info("=" * 60)
        logger.info("INICIANDO DISCOVERY DE PERFILES INSTAGRAM")
        logger.info("=" * 60)
        logger.info(f"Hashtags: {', '.join(self.config.hashtags)}")
        logger.info(f"Ubicación: {', '.join(self.config.location_keywords)}")
        logger.info(f"Nicho: {', '.join(self.config.niche_keywords)}")
        logger.info(f"Followers: {self.config.min_followers}-{self.config.max_followers}")
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
        """Extrae posts de todos los hashtags configurados."""
        all_posts = []

        for i, hashtag in enumerate(self.config.hashtags, 1):
            logger.info(f"[{i}/{len(self.config.hashtags)}] Procesando #{hashtag}")

            posts = self.client.scrape_hashtag_posts(
                hashtag=hashtag,
                max_posts=self.config.max_posts_per_hashtag,
                retries=self.config.max_retries,
                retry_delay=self.config.retry_delay,
            )

            for post in posts:
                post["_discovered_via_hashtag"] = hashtag

            all_posts.extend(posts)

            if i < len(self.config.hashtags):
                logger.info(f"Esperando {self.config.delay_between_requests}s...")
                time.sleep(self.config.delay_between_requests)

        logger.info(f"Total posts obtenidos: {len(all_posts)}")
        return all_posts

    def _extract_unique_usernames(self, posts: List[Dict[str, Any]]) -> None:
        """Extrae usernames únicos de los posts."""
        for post in posts:
            username = (
                post.get("ownerUsername") or
                post.get("username") or
                post.get("owner", {}).get("username")
            )

            if not username:
                continue

            username = username.lower().strip()

            if username in self.discovered_usernames:
                continue

            self.discovered_usernames.add(username)

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
        """Enriquece los perfiles con datos adicionales."""
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

                for detail in profile_details:
                    username = detail.get("username", "").lower().strip()
                    if username in self.profiles:
                        self._update_profile_from_details(username, detail)

            except Exception as e:
                logger.warning(f"Error enriqueciendo batch {batch_num + 1}: {e}")

            if batch_num < total_batches - 1:
                time.sleep(self.config.delay_between_requests)

    def _update_profile_from_details(self, username: str, details: Dict[str, Any]) -> None:
        """Actualiza un perfil con datos del profile scraper."""
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
        """Filtra perfiles por criterios y calcula puntuación de relevancia."""
        logger.info("Filtrando y puntuando perfiles...")

        relevant_profiles = []

        for username, profile in self.profiles.items():
            # Filtro de followers
            if profile.followers < self.config.min_followers:
                continue
            if profile.followers > self.config.max_followers:
                continue

            # Calcular relevancia
            relevance_score, location_detected, niche_detected = self._calculate_relevance(profile)

            profile.relevance_score = relevance_score
            profile.location_detected = location_detected
            profile.niche_detected = niche_detected

            # Filtro de relevancia mínima (al menos ubicación O nicho detectado)
            if not location_detected and not niche_detected:
                continue

            relevant_profiles.append(profile)

        # Ordenar por relevancia
        relevant_profiles.sort(key=lambda p: p.relevance_score, reverse=True)

        # Limitar output
        if len(relevant_profiles) > self.config.max_profiles_output:
            relevant_profiles = relevant_profiles[:self.config.max_profiles_output]

        logger.info(f"Perfiles relevantes: {len(relevant_profiles)}")
        return relevant_profiles

    def _calculate_relevance(self, profile: DiscoveredProfile) -> tuple[float, str, str]:
        """Calcula puntuación de relevancia."""
        score = 0.0
        location_detected = ""
        niche_detected = ""

        text_to_analyze = " ".join([
            profile.bio.lower(),
            profile.full_name.lower(),
            profile.category.lower(),
            profile.username.lower(),
        ])

        # Puntos por ubicación
        for keyword in self.config.location_keywords:
            if keyword.lower() in text_to_analyze:
                score += 30
                if not location_detected:
                    location_detected = keyword

        # Puntos por nicho
        niche_matches = []
        for keyword in self.config.niche_keywords:
            if keyword.lower() in text_to_analyze:
                score += 20
                niche_matches.append(keyword)

        if niche_matches:
            niche_detected = ", ".join(niche_matches[:3])

        # Bonus por cuenta de negocio
        if profile.is_business:
            score += 15

        # Bonus por categoría
        if profile.category:
            score += 10

        # Penalización por cuenta verificada (probablemente grande)
        if profile.is_verified:
            score -= 20

        # Bonus por rango óptimo de followers
        if 1000 <= profile.followers <= 15000:
            score += 10

        score = min(max(score, 0), 100)
        return score, location_detected, niche_detected

    def _save_results(self, profiles: List[DiscoveredProfile]) -> None:
        """Guarda resultados en CSV y JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.config.output_prefix}_{timestamp}"

        # CSV
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

        # JSON
        json_path = Path(self.config.output_dir) / f"{base_name}.json"

        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_profiles": len(profiles),
                "config": self.config.to_dict(),
            },
            "profiles": [p.to_dict() for p in profiles],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON guardado: {json_path}")

        # Resumen en consola
        print("\n" + "=" * 60)
        print("RESUMEN DE DISCOVERY")
        print("=" * 60)
        print(f"Total perfiles encontrados: {len(profiles)}")
        print(f"Archivos guardados en: {self.config.output_dir}/")
        print(f"  - {csv_path.name}")
        print(f"  - {json_path.name}")
        print("=" * 60)

        if profiles:
            print("\nTOP 10 PERFILES MAS RELEVANTES:")
            print("-" * 60)
            for i, p in enumerate(profiles[:10], 1):
                print(f"{i:2}. @{p.username:<25} | {p.followers:>6} seg | Score: {p.relevance_score:.0f}")
                if p.location_detected:
                    print(f"    Ubicacion: {p.location_detected}")
                if p.niche_detected:
                    print(f"    Nicho: {p.niche_detected}")
            print("-" * 60)


# =============================================================================
# CLI PARSER
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Descubre perfiles de Instagram relevantes para un nicho local.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Modo interactivo (pregunta qué buscar)
  python apify_instagram_discovery.py

  # Con archivo de configuración
  python apify_instagram_discovery.py --config mi_busqueda.yaml

  # Con argumentos directos
  python apify_instagram_discovery.py \\
      --hashtags "inmobiliariaalmeria,casasalmeria" \\
      --location "almería,roquetas" \\
      --niche "inmobiliaria,casas,pisos"

  # Guardar configuración para reutilizar
  python apify_instagram_discovery.py --save-config floristerias.yaml
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Archivo de configuración YAML/JSON'
    )

    parser.add_argument(
        '--hashtags',
        type=str,
        help='Hashtags separados por comas (sin #)'
    )

    parser.add_argument(
        '--location',
        type=str,
        help='Keywords de ubicación separados por comas'
    )

    parser.add_argument(
        '--niche',
        type=str,
        help='Keywords de nicho separados por comas'
    )

    parser.add_argument(
        '--min-followers',
        type=int,
        default=100,
        help='Mínimo de seguidores (default: 100)'
    )

    parser.add_argument(
        '--max-followers',
        type=int,
        default=50000,
        help='Máximo de seguidores (default: 50000)'
    )

    parser.add_argument(
        '--max-posts',
        type=int,
        default=150,
        help='Posts a buscar por hashtag (default: 150)'
    )

    parser.add_argument(
        '--max-profiles',
        type=int,
        default=150,
        help='Máximo perfiles en output (default: 150)'
    )

    parser.add_argument(
        '--no-enrich',
        action='store_true',
        help='Desactivar enriquecimiento de perfiles'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./discovery_output',
        help='Directorio de salida (default: ./discovery_output)'
    )

    parser.add_argument(
        '--save-config',
        type=str,
        help='Guardar configuración en archivo YAML para reutilizar'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Forzar modo interactivo'
    )

    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Optional[DiscoveryConfig]:
    """Construye configuración desde argumentos CLI."""
    if not args.hashtags:
        return None

    hashtags = [h.strip().lower().replace("#", "") for h in args.hashtags.split(",")]
    location = [l.strip().lower() for l in args.location.split(",")] if args.location else []
    niche = [n.strip().lower() for n in args.niche.split(",")] if args.niche else []

    return DiscoveryConfig(
        hashtags=hashtags,
        location_keywords=location,
        niche_keywords=niche,
        min_followers=args.min_followers,
        max_followers=args.max_followers,
        max_posts_per_hashtag=args.max_posts,
        max_profiles_output=args.max_profiles,
        enrich_profiles=not args.no_enrich,
        output_dir=args.output_dir,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Punto de entrada principal."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     APIFY INSTAGRAM DISCOVERY - Perfiles Locales          ║
    ║     Elena Redes / BrandPulse AI                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Verificar token
    if not os.environ.get("APIFY_API_TOKEN"):
        print("ERROR: APIFY_API_TOKEN no está configurado.")
        print("Configúralo con: export APIFY_API_TOKEN='tu_token_aquí'")
        sys.exit(1)

    # Parsear argumentos
    args = parse_arguments()

    # Determinar fuente de configuración
    config = None

    # 1. Archivo de configuración
    if args.config:
        if not Path(args.config).exists():
            print(f"ERROR: Archivo de configuración no encontrado: {args.config}")
            sys.exit(1)
        logger.info(f"Cargando configuración desde: {args.config}")
        config = DiscoveryConfig.from_yaml(args.config)

    # 2. Argumentos CLI
    elif args.hashtags:
        logger.info("Usando configuración desde argumentos CLI")
        config = build_config_from_args(args)

    # 3. Modo interactivo
    if config is None or args.interactive:
        config = run_interactive_setup()

    # Guardar configuración si se solicita
    if args.save_config:
        Path("./discovery_configs").mkdir(exist_ok=True)
        config_path = f"./discovery_configs/{args.save_config}"
        config.save_to_yaml(config_path)
        print(f"\nConfiguración guardada en: {config_path}")
        print("Puedes reutilizarla con: python apify_instagram_discovery.py --config " + config_path)

    # Ejecutar discovery
    try:
        engine = InstagramDiscoveryEngine(config)
        profiles = engine.run_discovery()

        if profiles:
            print(f"\nDiscovery completado!")
            print(f"{len(profiles)} perfiles relevantes encontrados.")
            print(f"Revisa los archivos en: {config.output_dir}/")
            print("\nSiguiente paso: Usa la extension Elena Bridge para")
            print("extraer el contenido de los perfiles que te interesen.")
        else:
            print("\nNo se encontraron perfiles relevantes.")
            print("Prueba con otros hashtags o ajusta los filtros.")

    except ValueError as e:
        print(f"\nError de configuración: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"\nDependencia faltante: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Error inesperado")
        print(f"\nError inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
