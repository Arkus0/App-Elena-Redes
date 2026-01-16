"""
Apify Service - Instagram & TikTok scraping
Primary data source for competitor analysis
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = logging.getLogger(__name__)


class ApifyService:
    """
    Service for scraping social media data via Apify API
    Supports Instagram, TikTok, and LinkedIn
    """

    def __init__(self):
        self.api_key = settings.APIFY_API_KEY
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Apify client if API key is available"""
        if self.api_key:
            try:
                from apify_client import ApifyClient
                self.client = ApifyClient(self.api_key)
                logger.info("Apify client initialized successfully")
            except ImportError:
                logger.warning("apify-client not installed, using mock data")
                self.client = None
        else:
            logger.warning("APIFY_API_KEY not set, using mock data")

    def is_available(self) -> bool:
        """Check if Apify is available"""
        return self.client is not None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def scrape_instagram_profile(
        self,
        username: str,
        max_posts: int = 30,
        include_reels: bool = True
    ) -> Dict[str, Any]:
        """
        Scrape Instagram profile and posts
        Returns profile info and top posts sorted by engagement
        """
        if not self.is_available():
            return self._get_mock_instagram_data(username)

        try:
            # Run the Instagram scraper actor
            run_input = {
                "directUrls": [f"https://www.instagram.com/{username}/"],
                "resultsType": "posts",
                "resultsLimit": max_posts,
                "searchType": "hashtag",
                "searchLimit": 1,
                "addParentData": True,
            }

            # Run actor and wait for completion
            run = self.client.actor(settings.APIFY_INSTAGRAM_ACTOR).call(run_input=run_input)

            # Get results
            items = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())

            if not items:
                logger.warning(f"No data returned for Instagram user {username}")
                return self._get_mock_instagram_data(username)

            # Process and structure the data
            return self._process_instagram_data(items, username)

        except Exception as e:
            logger.error(f"Error scraping Instagram {username}: {e}")
            return self._get_mock_instagram_data(username)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def scrape_tiktok_profile(
        self,
        username: str,
        max_videos: int = 30
    ) -> Dict[str, Any]:
        """
        Scrape TikTok profile and videos
        Returns profile info and top videos sorted by engagement
        """
        if not self.is_available():
            return self._get_mock_tiktok_data(username)

        try:
            run_input = {
                "profiles": [username],
                "resultsPerPage": max_videos,
                "shouldDownloadVideos": False,
                "shouldDownloadCovers": False,
            }

            run = self.client.actor(settings.APIFY_TIKTOK_ACTOR).call(run_input=run_input)
            items = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())

            if not items:
                logger.warning(f"No data returned for TikTok user {username}")
                return self._get_mock_tiktok_data(username)

            return self._process_tiktok_data(items, username)

        except Exception as e:
            logger.error(f"Error scraping TikTok {username}: {e}")
            return self._get_mock_tiktok_data(username)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def scrape_linkedin_profile(
        self,
        username: str,
        max_posts: int = 20
    ) -> Dict[str, Any]:
        """
        Scrape LinkedIn profile and posts
        """
        if not self.is_available():
            return self._get_mock_linkedin_data(username)

        try:
            run_input = {
                "profileUrls": [f"https://www.linkedin.com/in/{username}/"],
                "maxPosts": max_posts,
            }

            run = self.client.actor(settings.APIFY_LINKEDIN_ACTOR).call(run_input=run_input)
            items = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())

            if not items:
                return self._get_mock_linkedin_data(username)

            return self._process_linkedin_data(items, username)

        except Exception as e:
            logger.error(f"Error scraping LinkedIn {username}: {e}")
            return self._get_mock_linkedin_data(username)

    async def search_trending_content(
        self,
        keyword: str,
        platform: str = "instagram",
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for trending content by keyword
        Used for viral opportunity scanning
        """
        if not self.is_available():
            return self._get_mock_trending_data(keyword, platform)

        try:
            if platform == "instagram":
                run_input = {
                    "search": keyword,
                    "searchType": "hashtag",
                    "resultsLimit": max_results,
                }
                actor = settings.APIFY_INSTAGRAM_ACTOR
            else:  # TikTok
                run_input = {
                    "searchQueries": [keyword],
                    "resultsPerPage": max_results,
                }
                actor = settings.APIFY_TIKTOK_ACTOR

            run = self.client.actor(actor).call(run_input=run_input)
            items = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())

            return self._process_trending_data(items, platform)

        except Exception as e:
            logger.error(f"Error searching trending {platform} content for '{keyword}': {e}")
            return self._get_mock_trending_data(keyword, platform)

    def _process_instagram_data(self, items: List[Dict], username: str) -> Dict[str, Any]:
        """Process raw Instagram data into structured format"""
        profile_data = None
        posts = []

        for item in items:
            # Extract profile data from first item
            if not profile_data and "ownerUsername" in item:
                profile_data = {
                    "username": item.get("ownerUsername", username),
                    "full_name": item.get("ownerFullName", ""),
                    "bio": item.get("biography", ""),
                    "followers": item.get("followersCount", 0),
                    "following": item.get("followingCount", 0),
                    "posts_count": item.get("postsCount", 0),
                    "profile_pic": item.get("profilePicUrl", ""),
                }

            # Process post data
            post = {
                "platform_id": item.get("id", ""),
                "url": item.get("url", ""),
                "type": self._determine_instagram_type(item),
                "caption": item.get("caption", ""),
                "hashtags": item.get("hashtags", []),
                "mentions": item.get("mentions", []),
                "thumbnail": item.get("displayUrl", ""),
                "media_urls": item.get("images", []) or [item.get("displayUrl", "")],
                "video_duration": item.get("videoDuration"),
                "likes": item.get("likesCount", 0),
                "comments": item.get("commentsCount", 0),
                "video_views": item.get("videoViewCount", 0),
                "posted_at": item.get("timestamp"),
                "audio_name": item.get("musicInfo", {}).get("title") if item.get("musicInfo") else None,
            }

            # Calculate engagement score
            post["engagement_score"] = self._calculate_engagement_score(
                post["likes"],
                post["comments"],
                0,  # saves not available directly
                0,  # shares not available directly
                post["video_views"]
            )

            posts.append(post)

        # Sort by engagement
        posts.sort(key=lambda x: x["engagement_score"], reverse=True)

        return {
            "profile": profile_data or {"username": username},
            "posts": posts[:30],  # Top 30
            "scraped_at": datetime.utcnow().isoformat(),
        }

    def _process_tiktok_data(self, items: List[Dict], username: str) -> Dict[str, Any]:
        """Process raw TikTok data into structured format"""
        profile_data = None
        videos = []

        for item in items:
            if not profile_data and "authorMeta" in item:
                author = item["authorMeta"]
                profile_data = {
                    "username": author.get("name", username),
                    "display_name": author.get("nickName", ""),
                    "bio": author.get("signature", ""),
                    "followers": author.get("fans", 0),
                    "following": author.get("following", 0),
                    "likes": author.get("heart", 0),
                    "videos_count": author.get("video", 0),
                    "profile_pic": author.get("avatar", ""),
                }

            video = {
                "platform_id": item.get("id", ""),
                "url": item.get("webVideoUrl", ""),
                "type": "tiktok_video",
                "caption": item.get("text", ""),
                "hashtags": [h.get("name", "") for h in item.get("hashtags", [])],
                "mentions": item.get("mentions", []),
                "thumbnail": item.get("covers", {}).get("default", ""),
                "video_duration": item.get("videoMeta", {}).get("duration", 0),
                "likes": item.get("diggCount", 0),
                "comments": item.get("commentCount", 0),
                "shares": item.get("shareCount", 0),
                "plays": item.get("playCount", 0),
                "saves": item.get("collectCount", 0),
                "posted_at": item.get("createTime"),
                "audio_name": item.get("musicMeta", {}).get("musicName"),
                "audio_author": item.get("musicMeta", {}).get("musicAuthor"),
                "audio_original": item.get("musicMeta", {}).get("musicOriginal", False),
            }

            video["engagement_score"] = self._calculate_engagement_score(
                video["likes"],
                video["comments"],
                video["saves"],
                video["shares"],
                video["plays"]
            )

            videos.append(video)

        videos.sort(key=lambda x: x["engagement_score"], reverse=True)

        return {
            "profile": profile_data or {"username": username},
            "posts": videos[:30],
            "scraped_at": datetime.utcnow().isoformat(),
        }

    def _process_linkedin_data(self, items: List[Dict], username: str) -> Dict[str, Any]:
        """Process raw LinkedIn data"""
        profile_data = None
        posts = []

        for item in items:
            if not profile_data:
                profile_data = {
                    "username": username,
                    "full_name": item.get("authorName", ""),
                    "headline": item.get("authorHeadline", ""),
                    "followers": item.get("authorFollowersCount", 0),
                }

            post = {
                "platform_id": item.get("urn", ""),
                "url": item.get("postUrl", ""),
                "type": "linkedin_post" if not item.get("document") else "linkedin_carousel",
                "caption": item.get("text", ""),
                "likes": item.get("numLikes", 0),
                "comments": item.get("numComments", 0),
                "shares": item.get("numShares", 0),
                "posted_at": item.get("postedAt"),
            }

            post["engagement_score"] = self._calculate_engagement_score(
                post["likes"], post["comments"], 0, post["shares"], 0
            )

            posts.append(post)

        posts.sort(key=lambda x: x["engagement_score"], reverse=True)

        return {
            "profile": profile_data or {"username": username},
            "posts": posts[:20],
            "scraped_at": datetime.utcnow().isoformat(),
        }

    def _process_trending_data(self, items: List[Dict], platform: str) -> List[Dict[str, Any]]:
        """Process trending content search results"""
        results = []
        for item in items:
            if platform == "instagram":
                results.append({
                    "platform_id": item.get("id", ""),
                    "url": item.get("url", ""),
                    "caption": item.get("caption", ""),
                    "likes": item.get("likesCount", 0),
                    "comments": item.get("commentsCount", 0),
                    "views": item.get("videoViewCount", 0),
                    "thumbnail": item.get("displayUrl", ""),
                    "author": item.get("ownerUsername", ""),
                })
            else:
                results.append({
                    "platform_id": item.get("id", ""),
                    "url": item.get("webVideoUrl", ""),
                    "caption": item.get("text", ""),
                    "likes": item.get("diggCount", 0),
                    "comments": item.get("commentCount", 0),
                    "plays": item.get("playCount", 0),
                    "shares": item.get("shareCount", 0),
                    "thumbnail": item.get("covers", {}).get("default", ""),
                    "author": item.get("authorMeta", {}).get("name", ""),
                })

        return sorted(results, key=lambda x: x.get("likes", 0) + x.get("comments", 0), reverse=True)

    def _determine_instagram_type(self, item: Dict) -> str:
        """Determine Instagram content type"""
        if item.get("type") == "Video" or item.get("isVideo"):
            return "reel"
        elif item.get("type") == "Sidecar" or len(item.get("images", [])) > 1:
            return "carousel"
        else:
            return "static_image"

    def _calculate_engagement_score(
        self,
        likes: int,
        comments: int,
        saves: int,
        shares: int,
        views: int
    ) -> float:
        """
        Calculate weighted engagement score
        Comments > Saves > Shares > Likes (intent hierarchy)
        """
        weighted_engagement = (
            likes +
            (comments * 3) +
            (saves * 5) +
            (shares * 4)
        )

        # Normalize by views if available
        if views > 0:
            engagement_rate = (weighted_engagement / views) * 100
            return min(engagement_rate * 10, 100)

        # Otherwise use raw score with log scaling
        import math
        if weighted_engagement > 0:
            return min(math.log10(weighted_engagement + 1) * 20, 100)
        return 0

    # ============ MOCK DATA FOR DEMO ============

    def _get_mock_instagram_data(self, username: str) -> Dict[str, Any]:
        """Generate realistic mock data for Instagram demo"""
        return {
            "profile": {
                "username": username,
                "full_name": f"{username.replace('_', ' ').title()}",
                "bio": "Tu floristeria de confianza | Ramos personalizados | Envios a domicilio | Barcelona",
                "followers": 12500,
                "following": 890,
                "posts_count": 342,
            },
            "posts": self._generate_mock_instagram_posts(username),
            "scraped_at": datetime.utcnow().isoformat(),
        }

    def _get_mock_tiktok_data(self, username: str) -> Dict[str, Any]:
        """Generate realistic mock data for TikTok demo"""
        return {
            "profile": {
                "username": username,
                "display_name": username.replace("_", " ").title(),
                "bio": "Floristeria artesanal | Tutoriales de arreglos | Barcelona",
                "followers": 45000,
                "following": 234,
                "likes": 890000,
                "videos_count": 156,
            },
            "posts": self._generate_mock_tiktok_videos(username),
            "scraped_at": datetime.utcnow().isoformat(),
        }

    def _get_mock_linkedin_data(self, username: str) -> Dict[str, Any]:
        """Generate mock LinkedIn data"""
        return {
            "profile": {
                "username": username,
                "full_name": username.replace("_", " ").title(),
                "headline": "CEO @ Floristeria Artesanal | Emprendedor | Sostenibilidad",
                "followers": 5600,
            },
            "posts": self._generate_mock_linkedin_posts(username),
            "scraped_at": datetime.utcnow().isoformat(),
        }

    def _get_mock_trending_data(self, keyword: str, platform: str) -> List[Dict[str, Any]]:
        """Generate mock trending data"""
        trending_items = [
            {
                "platform_id": f"trend_{i}",
                "url": f"https://{platform}.com/trending/{i}",
                "caption": f"Trending content about {keyword} #{keyword.replace(' ', '')} #viral #fyp",
                "likes": 15000 - (i * 1000),
                "comments": 500 - (i * 30),
                "views": 150000 - (i * 10000),
                "plays": 150000 - (i * 10000),
                "shares": 200 - (i * 15),
                "thumbnail": f"https://example.com/thumb_{i}.jpg",
                "author": f"trending_creator_{i}",
            }
            for i in range(10)
        ]
        return trending_items

    def _generate_mock_instagram_posts(self, username: str) -> List[Dict[str, Any]]:
        """Generate realistic Instagram posts for floristeria demo"""
        posts = [
            {
                "platform_id": "ig_001",
                "url": f"https://instagram.com/p/abc123",
                "type": "reel",
                "caption": "POV: Cuando el cliente dice 'sorprendeme' y le entregas ESTO\n\nEste ramo de peonias con eucalipto fue todo un exito\n\nGuarda este video si quieres ideas para tu proximo regalo\n\n#floristeria #ramosdenovias #peonias #barcelona #flores",
                "hashtags": ["floristeria", "ramosdenovias", "peonias", "barcelona", "flores"],
                "mentions": [],
                "thumbnail": "https://example.com/thumb1.jpg",
                "video_duration": 28,
                "likes": 8500,
                "comments": 342,
                "video_views": 125000,
                "posted_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
                "audio_name": "Flowers - Miley Cyrus",
                "engagement_score": 92.5,
            },
            {
                "platform_id": "ig_002",
                "url": f"https://instagram.com/p/def456",
                "type": "carousel",
                "caption": "5 errores que TODOS cometen al cuidar sus flores en casa\n\n1. Demasiada agua (las raices se pudren)\n2. Sol directo (quema las hojas)\n3. No cortar los tallos en angulo\n4. Usar agua muy fria\n5. No cambiar el agua cada 2 dias\n\nComenta cual has cometido tu\n\n#tipsflores #cuidadodeflores #plantlovers",
                "hashtags": ["tipsflores", "cuidadodeflores", "plantlovers"],
                "mentions": [],
                "thumbnail": "https://example.com/thumb2.jpg",
                "video_duration": None,
                "likes": 5200,
                "comments": 487,
                "video_views": 0,
                "posted_at": (datetime.utcnow() - timedelta(days=7)).isoformat(),
                "audio_name": None,
                "engagement_score": 88.3,
            },
            {
                "platform_id": "ig_003",
                "url": f"https://instagram.com/p/ghi789",
                "type": "reel",
                "caption": "De esto a ESTO en 3 minutos\n\nTransformacion de ramo basico a ramo premium\n\nEl secreto? Añadir textura con eucalipto y gypsophila\n\nDM si quieres aprender la tecnica completa\n\n#transformacion #floristeria #tutorialflores",
                "hashtags": ["transformacion", "floristeria", "tutorialflores"],
                "mentions": [],
                "thumbnail": "https://example.com/thumb3.jpg",
                "video_duration": 22,
                "likes": 12300,
                "comments": 567,
                "video_views": 234000,
                "posted_at": (datetime.utcnow() - timedelta(days=14)).isoformat(),
                "audio_name": "original sound",
                "engagement_score": 95.8,
            },
            {
                "platform_id": "ig_004",
                "url": f"https://instagram.com/p/jkl012",
                "type": "reel",
                "caption": "Un dia en mi floristeria (todo lo que no ves en Instagram)\n\n5:30 - Mercado de flores\n7:00 - Preparar el local\n8:00 - Primeros pedidos\n...\n20:00 - Cerrar agotada pero feliz\n\nEsto es emprender\n\n#behindthescenes #emprender #floristeria #diaaadia",
                "hashtags": ["behindthescenes", "emprender", "floristeria", "diaaadia"],
                "mentions": [],
                "thumbnail": "https://example.com/thumb4.jpg",
                "video_duration": 45,
                "likes": 6800,
                "comments": 423,
                "video_views": 98000,
                "posted_at": (datetime.utcnow() - timedelta(days=21)).isoformat(),
                "audio_name": "Daylight - David Kushner",
                "engagement_score": 85.2,
            },
            {
                "platform_id": "ig_005",
                "url": f"https://instagram.com/p/mno345",
                "type": "static_image",
                "caption": "Nuevo en la tienda: Ramos preservados que duran 2+ años\n\nPerfecto para:\n- Regalo duradero\n- Decoracion de hogar\n- Bodas (ramo de novia eterno)\n\nDM 'PRESERVADO' y te enviamos catalogo\n\n#ramospreservados #florespreservadas #regaloespecial",
                "hashtags": ["ramospreservados", "florespreservadas", "regaloespecial"],
                "mentions": [],
                "thumbnail": "https://example.com/thumb5.jpg",
                "video_duration": None,
                "likes": 3400,
                "comments": 156,
                "video_views": 0,
                "posted_at": (datetime.utcnow() - timedelta(days=28)).isoformat(),
                "audio_name": None,
                "engagement_score": 72.4,
            },
        ]

        # Add more varied posts
        additional_hooks = [
            "3 flores que NUNCA debes regalar (la #2 te sorprendera)",
            "Cliente: 'Es para pedir perdon' | Yo: Di no mas",
            "El ramo mas dificil que he hecho en 10 anos",
            "Cuanto cuesta abrir una floristeria? (numeros reales)",
            "Tendencias florales 2026 que debes conocer",
        ]

        for i, hook in enumerate(additional_hooks):
            posts.append({
                "platform_id": f"ig_{i+6:03d}",
                "url": f"https://instagram.com/p/post{i+6}",
                "type": "reel" if i % 2 == 0 else "carousel",
                "caption": f"{hook}\n\n#floristeria #flores #tendencias",
                "hashtags": ["floristeria", "flores", "tendencias"],
                "mentions": [],
                "thumbnail": f"https://example.com/thumb{i+6}.jpg",
                "video_duration": 25 if i % 2 == 0 else None,
                "likes": 4000 - (i * 300),
                "comments": 200 - (i * 20),
                "video_views": 80000 - (i * 5000) if i % 2 == 0 else 0,
                "posted_at": (datetime.utcnow() - timedelta(days=35 + i*7)).isoformat(),
                "audio_name": "trending sound" if i % 2 == 0 else None,
                "engagement_score": 75 - (i * 3),
            })

        return posts

    def _generate_mock_tiktok_videos(self, username: str) -> List[Dict[str, Any]]:
        """Generate realistic TikTok videos for demo"""
        videos = [
            {
                "platform_id": "tt_001",
                "url": f"https://tiktok.com/@{username}/video/001",
                "type": "tiktok_video",
                "caption": "Respuesta a @maria: Como hacer un ramo en 60 segundos #flores #tutorial #floristeria #fyp",
                "hashtags": ["flores", "tutorial", "floristeria", "fyp"],
                "mentions": ["maria"],
                "thumbnail": "https://example.com/tt_thumb1.jpg",
                "video_duration": 58,
                "likes": 45000,
                "comments": 890,
                "shares": 2300,
                "plays": 890000,
                "saves": 12000,
                "posted_at": (datetime.utcnow() - timedelta(days=5)).isoformat(),
                "audio_name": "Espresso - Sabrina Carpenter",
                "audio_original": False,
                "engagement_score": 97.2,
            },
            {
                "platform_id": "tt_002",
                "url": f"https://tiktok.com/@{username}/video/002",
                "type": "tiktok_video",
                "caption": "El PEOR cliente que he tenido (storytime) #floristeria #storytime #emprender",
                "hashtags": ["floristeria", "storytime", "emprender"],
                "mentions": [],
                "thumbnail": "https://example.com/tt_thumb2.jpg",
                "video_duration": 89,
                "likes": 67000,
                "comments": 3400,
                "shares": 5600,
                "plays": 1200000,
                "saves": 8900,
                "posted_at": (datetime.utcnow() - timedelta(days=12)).isoformat(),
                "audio_name": "original sound",
                "audio_original": True,
                "engagement_score": 98.5,
            },
            {
                "platform_id": "tt_003",
                "url": f"https://tiktok.com/@{username}/video/003",
                "type": "tiktok_video",
                "caption": "POV: Aprendes a hacer ramos viendo TikToks #aprender #flores #diy",
                "hashtags": ["aprender", "flores", "diy"],
                "mentions": [],
                "thumbnail": "https://example.com/tt_thumb3.jpg",
                "video_duration": 34,
                "likes": 23000,
                "comments": 567,
                "shares": 1200,
                "plays": 450000,
                "saves": 15000,
                "posted_at": (datetime.utcnow() - timedelta(days=20)).isoformat(),
                "audio_name": "Oh No - Kreepa",
                "audio_original": False,
                "engagement_score": 91.3,
            },
        ]

        return videos

    def _generate_mock_linkedin_posts(self, username: str) -> List[Dict[str, Any]]:
        """Generate realistic LinkedIn posts for demo"""
        return [
            {
                "platform_id": "li_001",
                "url": f"https://linkedin.com/posts/{username}_001",
                "type": "linkedin_carousel",
                "caption": "5 lecciones de emprendimiento que aprendi en mi floristeria (y que no te ensenan en ninguna escuela de negocios)\n\n1. El producto es solo el 20% del exito\n2. Tus primeros clientes son tu mejor marketing\n3. Reinvierte antes de pagar mas\n4. La comunidad local es tu mayor activo\n5. Diferenciarte no significa ser el mas caro\n\nCual agregarías tu?\n\n#emprendimiento #pymes #negocios #floristeria",
                "likes": 2300,
                "comments": 156,
                "shares": 89,
                "posted_at": (datetime.utcnow() - timedelta(days=10)).isoformat(),
                "engagement_score": 82.5,
            },
        ]
