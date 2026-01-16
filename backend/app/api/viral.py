"""
Viral Scanner API Routes
Scan trending content and generate reactive ideas
"""
from typing import List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.business import Business
from app.schemas.content import (
    ViralScanRequest,
    ViralOpportunity,
    ViralScanResponse,
)
from app.services.content_generator import ContentGenerator

router = APIRouter()
content_generator = ContentGenerator()


@router.post("/{business_id}/scan", response_model=ViralScanResponse)
async def scan_viral_opportunities(
    business_id: int,
    request: ViralScanRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Scan for viral opportunities based on keyword/trend
    Returns actionable content ideas based on what's trending
    """
    # Verify business access
    biz_result = await db.execute(
        select(Business)
        .where(Business.id == business_id)
        .where(Business.user_id == current_user.id)
    )
    business = biz_result.scalar_one_or_none()
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")

    # Scan for viral opportunities
    ideas = await content_generator.scan_viral_opportunities(
        db=db,
        business_id=business_id,
        keyword=request.keyword,
        platform=request.platform,
    )

    # Build response
    opportunities = []
    for idea in ideas:
        opportunities.append(ViralOpportunity(
            trend_id=f"trend_{hash(idea.get('idea_title', ''))}"[:10],
            trend_name=idea.get("idea_title", ""),
            platform=idea.get("platform", request.platform),
            description=idea.get("script_outline", [""])[0] if isinstance(idea.get("script_outline"), list) else "",
            total_views=0,  # Would come from Apify in real implementation
            total_videos=0,
            growth_rate="rising",
            time_sensitive=idea.get("difficulty") == "facil",
            relevance_score=idea.get("relevance_score", 75),
            difficulty_score={"facil": 30, "medio": 60, "dificil": 85}.get(idea.get("difficulty", "medio"), 60),
            content_ideas=[{
                "title": idea.get("idea_title"),
                "hook": idea.get("hook"),
                "script_outline": idea.get("script_outline"),
                "filming_tips": idea.get("filming_tips"),
            }],
            top_examples=[],
            best_time_to_post="Entre 11:00-13:00 o 19:00-21:00",
            trend_expiry_estimate="5-7 días" if idea.get("difficulty") == "facil" else "2-3 semanas",
        ))

    return ViralScanResponse(
        keyword=request.keyword,
        platform=request.platform,
        scan_date=datetime.utcnow(),
        opportunities=opportunities,
        general_insights=[
            f"Encontramos {len(opportunities)} oportunidades relacionadas con '{request.keyword}'",
            "Las tendencias con 'POV' y 'storytime' están funcionando muy bien ahora",
            "El contenido educativo tipo 'tips' tiene engagement consistente",
            "Actuar rápido en tendencias (primeras 48-72h) multiplica el alcance",
        ],
    )


@router.get("/{business_id}/trending")
async def get_trending_in_niche(
    business_id: int,
    platform: str = "instagram",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get current trending content in the business's niche
    """
    # Verify business access
    biz_result = await db.execute(
        select(Business)
        .where(Business.id == business_id)
        .where(Business.user_id == current_user.id)
    )
    business = biz_result.scalar_one_or_none()
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")

    # Get niche-specific trends
    niche_keywords = {
        "floristeria": ["flores", "ramos", "arreglos florales", "florist"],
        "inmobiliaria": ["casas", "inmuebles", "real estate", "pisos"],
        "cafeteria": ["cafe", "coffee", "barista", "latte art"],
        "peluqueria": ["pelo", "hair", "cortes", "peinados"],
        "tienda_local": ["tienda", "local", "compra local", "shop small"],
        "restaurante": ["comida", "food", "recetas", "cocina"],
        "gimnasio": ["fitness", "gym", "workout", "ejercicio"],
    }

    keywords = niche_keywords.get(business.business_type.value, ["trending", "viral"])

    # In a real implementation, this would call Apify to get actual trending content
    trending_data = {
        "platform": platform,
        "business_niche": business.business_type.value,
        "trending_now": [
            {
                "trend": "POV content",
                "description": "Videos en primera persona que crean identificación",
                "example_hook": f"POV: Trabajas en una {business.business_type.value}",
                "engagement_potential": "muy alto",
                "difficulty": "fácil",
            },
            {
                "trend": "Before/After transformations",
                "description": "Contenido de transformación visual",
                "example_hook": "De esto... a ESTO",
                "engagement_potential": "muy alto",
                "difficulty": "fácil",
            },
            {
                "trend": "Day in my life",
                "description": "Un día en la vida de tu negocio",
                "example_hook": "5am: Lo que nadie ve de emprender",
                "engagement_potential": "alto",
                "difficulty": "medio",
            },
            {
                "trend": "Tips rápidos",
                "description": "Consejos prácticos en menos de 30 segundos",
                "example_hook": f"3 cosas que NO sabías sobre {keywords[0]}",
                "engagement_potential": "alto",
                "difficulty": "fácil",
            },
            {
                "trend": "Storytime",
                "description": "Historias personales con gancho emocional",
                "example_hook": "El cliente que nunca olvidaré...",
                "engagement_potential": "muy alto",
                "difficulty": "medio",
            },
        ],
        "trending_audios": [
            {"name": "Espresso - Sabrina Carpenter", "type": "upbeat, trendy"},
            {"name": "Die With A Smile - Lady Gaga", "type": "emotional"},
            {"name": "Please Please Please - Sabrina Carpenter", "type": "catchy"},
            {"name": "Original audio con voiceover", "type": "authentic"},
        ],
        "recommendations": [
            "Publica contenido de tendencia en las primeras 48-72 horas para máximo alcance",
            "Combina trending audio con contenido de valor para mejor retención",
            "Los hooks con preguntas o sorpresas tienen 2x más views",
        ],
    }

    return trending_data


@router.post("/{business_id}/quick-idea")
async def generate_quick_viral_idea(
    business_id: int,
    trend_type: str = "pov",  # pov, transformation, tips, storytime, dayinlife
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate a quick viral idea based on trending format
    """
    # Verify business access
    biz_result = await db.execute(
        select(Business)
        .where(Business.id == business_id)
        .where(Business.user_id == current_user.id)
    )
    business = biz_result.scalar_one_or_none()
    if not business:
        raise HTTPException(status_code=404, detail="Business not found")

    # Quick idea templates by trend type
    templates = {
        "pov": {
            "hook": f"POV: Trabajas en una {business.business_type.value} y un cliente te pide algo imposible",
            "structure": [
                "0-3s: Cara de sorpresa mirando a cámara + texto del hook",
                "3-15s: Mostrar la situación 'imposible' con humor",
                "15-25s: Resolver creativamente (timelapse o corte)",
                "25-30s: Reveal final + CTA '¿Te ha pasado?'",
            ],
            "filming_tips": "Expresiones faciales exageradas, cortes rápidos, buena iluminación frontal",
            "suggested_audio": "Oh No - Kreepa",
            "difficulty": "fácil",
            "time_to_create": "20 minutos",
        },
        "transformation": {
            "hook": "De ESTO... a esto",
            "structure": [
                "0-2s: Mostrar estado inicial (poco atractivo)",
                "2-3s: Texto 'Watch this' o transición",
                "3-20s: Timelapse de transformación",
                "20-25s: Reveal dramático del resultado",
                "25-30s: Comparación side-by-side + CTA",
            ],
            "filming_tips": "Mismo ángulo para before/after, buena luz, música épica en el reveal",
            "suggested_audio": "Build up + drop trending",
            "difficulty": "fácil",
            "time_to_create": "30 minutos",
        },
        "tips": {
            "hook": f"3 cosas que NADIE te dice sobre {business.business_type.value}",
            "structure": [
                "0-3s: Hook con número + expresión de 'te voy a contar un secreto'",
                "3-10s: Tip 1 (el más sorprendente)",
                "10-18s: Tip 2 con demostración visual",
                "18-25s: Tip 3",
                "25-30s: CTA '¿Cuál no sabías? Comenta'",
            ],
            "filming_tips": "Números grandes en pantalla, transiciones rápidas, energía alta",
            "suggested_audio": "Audio upbeat o voiceover",
            "difficulty": "fácil",
            "time_to_create": "25 minutos",
        },
        "storytime": {
            "hook": "El cliente que nunca voy a olvidar...",
            "structure": [
                "0-3s: Hook emocional mirando a cámara",
                "3-15s: Contexto de la historia",
                "15-22s: El momento clave/twist",
                "22-28s: Resolución emocional",
                "28-30s: CTA '¿Te ha pasado algo similar?'",
            ],
            "filming_tips": "Close-up de cara, tono conversacional, pausas dramáticas",
            "suggested_audio": "Audio original o música suave de fondo",
            "difficulty": "medio",
            "time_to_create": "15 minutos",
        },
        "dayinlife": {
            "hook": f"Un día real trabajando en {business.name}",
            "structure": [
                "0-3s: Hora temprana + 'Vamos'",
                "3-10s: Preparación/apertura",
                "10-20s: Momentos del día (clientes, trabajo)",
                "20-25s: Momento especial o logro",
                "25-30s: Cierre + '¿Quieres ver más días?'",
            ],
            "filming_tips": "Clips cortos de 2-3 segundos, mostrar variedad, música motivacional",
            "suggested_audio": "Trending motivational audio",
            "difficulty": "medio",
            "time_to_create": "Filmar durante el día + 20 min edición",
        },
    }

    idea = templates.get(trend_type, templates["pov"])

    return {
        "trend_type": trend_type,
        "business": business.name,
        "idea": idea,
        "hashtags": [
            f"#{business.business_type.value}",
            "#emprender",
            "#negociolocal",
            f"#{business.location.lower().replace(' ', '') if business.location else 'españa'}",
            "#fyp",
            "#viral",
        ],
        "best_posting_times": ["11:00", "13:00", "19:00", "21:00"],
        "ready_to_film": True,
    }
