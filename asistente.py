"""
SustentIA — Asistente Inteligente con Llama (Hackathon 1).

Asistente conversacional para la inmobiliaria Sustentia (Morelia, Michoacán) con
arquitectura de continuidad en tres niveles sobre la interfaz compatible con
OpenAI:
1. Proveedor principal: Meta Llama (`meta-llama/llama-3.1-8b-instruct`, con
   alternativa `meta-llama/llama-3.3-70b-instruct` de mayor fidelidad al prompt)
   vía OpenRouter (`https://openrouter.ai/api/v1`).
2. Respaldo automático: Groq (`https://api.groq.com/openai/v1`) con el modelo
   `openai/gpt-oss-120b`. Nota importante de transparencia: `openai/gpt-oss-120b`
   NO es un modelo Llama; su función es garantizar la continuidad operativa para
   que el asistente nunca deje de responder en producción si OpenRouter falla o
   agota sus créditos.
3. Último recurso: modo `[SIMULADO]` en Python local.

Atiende prospectos fuera de horario, identifica necesidad y presupuesto,
recomienda desarrollos del catálogo sin inventar datos, recuerda la conversación,
captura datos del prospecto y escala a un asesor humano cuando corresponde.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
import unicodedata

# Importación opcional del SDK de OpenAI (compatible con OpenRouter y Groq):
# permite ejecutar el asistente en modo [SIMULADO] aun cuando el paquete
# `openai` no esté instalado o no existan API keys.
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ==============================================================================
# 1. CONFIGURACIÓN GLOBAL Y CATÁLOGO OFICIAL DE DESARROLLOS
# ==============================================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODELO_LLAMA = "meta-llama/llama-3.1-8b-instruct"
# Modelo alternativo con mayor fidelidad a las reglas del prompt (verificado en pruebas):
# MODELO_LLAMA = "meta-llama/llama-3.3-70b-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODELO_RESPALDO_GROQ = "openai/gpt-oss-120b"
CREDITOS_ACEPTADOS = ["Infonavit", "Fovissste", "Bancario"]
CONTACTO_ASESOR = "[teléfono del asesor]"
MENSAJE_ESCALAMIENTO = "Te pongo en contacto con un asesor, un momento, por favor."

# Fuente única de verdad del catálogo de Sustentia.
# Regla estricta: solo zonas generales, sin direcciones exactas ni inventarios de unidades.
CATALOGO_DESARROLLOS = [
    {
        "nombre": "San Pedro",
        "tipo": "casas",
        "zona": "Morelia, zona Walmart La Huerta",
        "precio_desde": 1250000,
        "precio_texto": "desde $1,250,000",
        "recamaras": "2 a 3 recámaras",
        "detalles": "2 a 3 recámaras; entrega inmediata en algunos modelos",
        "creditos": CREDITOS_ACEPTADOS,
    },
    {
        "nombre": "Torres Zojo",
        "tipo": "departamentos",
        "zona": "Morelia, por el Tecnológico de Morelia",
        "precio_desde": 1200000,
        "precio_texto": "desde $1,200,000 aprox.",
        "recamaras": None,
        "detalles": "Departamentos ubicados por el Tecnológico de Morelia",
        "creditos": CREDITOS_ACEPTADOS,
    },
    {
        "nombre": "Veronesa",
        "tipo": "casas",
        "zona": "Morelia, zona La Calera",
        "precio_desde": 2690000,
        "precio_texto": "desde $2,690,000",
        "recamaras": "3 recámaras",
        "detalles": "3 recámaras",
        "creditos": CREDITOS_ACEPTADOS,
    },
    {
        "nombre": "Paseo de Andalucía (modelos Alba y Leira)",
        "tipo": "departamentos y casas",
        "zona": "Morelia, zona Altozano",
        "precio_desde": 2830000,
        "precio_texto": "desde $2,830,000",
        "recamaras": None,
        "detalles": "Modelos Alba y Leira (departamentos y casas)",
        "creditos": CREDITOS_ACEPTADOS,
    },
    {
        "nombre": "Rincón de las Aves",
        "tipo": "casas",
        "zona": "Morelia, zona El Vergel",
        "precio_desde": 2245000,
        "precio_texto": "desde $2,245,000",
        "recamaras": None,
        "detalles": "Casa club y gimnasio",
        "creditos": CREDITOS_ACEPTADOS,
    },
    {
        "nombre": "Tepatzícuaro",
        "tipo": "lotes residenciales campestres",
        "zona": "Acuitzio del Canje, a 20 min de Morelia",
        "precio_desde": 705127,
        "precio_texto": "desde $705,127",
        "recamaras": None,
        "detalles": "Lago, trotapista y ciclovía",
        "creditos": CREDITOS_ACEPTADOS,
    },
    {
        "nombre": "Valle Verde",
        "tipo": "lotes",
        "zona": "Morelia",
        "precio_desde": None,
        "precio_por_m2": 5000,
        "precio_texto": "$5,000 por m²",
        "recamaras": None,
        "detalles": "Lotes en Morelia con precio por metro cuadrado",
        "creditos": CREDITOS_ACEPTADOS,
    },
]


# ==============================================================================
# 2. CONEXIÓN SEGURA: OPENROUTER (PRINCIPAL) Y GROQ (RESPALDO)
# ==============================================================================


def _obtener_api_key_secreto(
    nombre_secreto: str,
    nombre_proveedor: str,
    permitir_prompt: bool = True,
) -> str:
    """
    Obtiene una API key de forma segura sin exponerla en el código.

    Orden de búsqueda:
    1. Secretos de Google Colab (`google.colab.userdata`).
    2. Variable de entorno `nombre_secreto`.
    3. Entrada interactiva oculta con `getpass` (si está en Colab o terminal TTY).
    """
    try:
        from google.colab import userdata  # type: ignore

        clave_colab = userdata.get(nombre_secreto)
        if clave_colab and clave_colab.strip():
            return clave_colab.strip()
    except Exception:
        pass

    clave_entorno = os.environ.get(nombre_secreto, "").strip()
    if clave_entorno:
        return clave_entorno

    if (
        permitir_prompt
        and os.environ.get("SUSTENTIA_MODO_SIMULADO") != "1"
        and ("google.colab" in sys.modules or sys.stdin.isatty())
    ):
        try:
            clave_manual = getpass.getpass(
                f"Ingresa tu API key de {nombre_proveedor} ({nombre_secreto}, o presiona Enter para omitir): "
            )
            return clave_manual.strip()
        except (EOFError, KeyboardInterrupt, Exception):
            return ""

    return ""


def obtener_api_key(permitir_prompt: bool = True) -> str:
    """Obtiene la API key principal de OpenRouter (`OPENROUTER_API_KEY`)."""
    return _obtener_api_key_secreto(
        "OPENROUTER_API_KEY",
        "OpenRouter",
        permitir_prompt=permitir_prompt,
    )


def obtener_api_key_groq(permitir_prompt: bool = True) -> str:
    """Obtiene la API key de respaldo de Groq (`GROQ_API_KEY`)."""
    return _obtener_api_key_secreto(
        "GROQ_API_KEY",
        "Groq (respaldo)",
        permitir_prompt=permitir_prompt,
    )


def crear_cliente_openrouter(api_key: str | None = None):
    """
    Inicializa el cliente principal de OpenRouter (`https://openrouter.ai/api/v1`)
    vía SDK `openai` si el paquete y la clave están disponibles.
    Devuelve `None` cuando no hay llave de OpenRouter.
    """
    if OpenAI is None:
        return None
    clave = api_key if api_key is not None else obtener_api_key(permitir_prompt=False)
    if not clave:
        return None
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=clave)


def crear_cliente_groq(api_key: str | None = None):
    """
    Inicializa el cliente de respaldo en Groq (`https://api.groq.com/openai/v1`)
    vía SDK `openai` si el paquete y la clave están disponibles.
    Devuelve `None` cuando no hay llave de Groq.
    """
    if OpenAI is None:
        return None
    clave = api_key if api_key is not None else obtener_api_key_groq(permitir_prompt=False)
    if not clave:
        return None
    return OpenAI(base_url=GROQ_BASE_URL, api_key=clave)


def describir_proveedor_activo(
    cliente_openrouter,
    cliente_groq,
    forzar_simulado: bool = False,
) -> str:
    """Describe de forma clara el proveedor principal y respaldo disponibles."""
    if forzar_simulado:
        return "[SIMULADO]"
    if cliente_openrouter is None and cliente_groq is None:
        return "[SIMULADO] (sin llaves configuradas)"
    if cliente_openrouter is not None and cliente_groq is not None:
        return (
            f"OpenRouter ({MODELO_LLAMA}) + respaldo automático Groq ({MODELO_RESPALDO_GROQ})"
        )
    if cliente_openrouter is not None:
        return f"OpenRouter ({MODELO_LLAMA}) [sin respaldo Groq; último recurso: [SIMULADO]]"
    return f"Groq ({MODELO_RESPALDO_GROQ}) [OpenRouter omitido; último recurso: [SIMULADO]]"


# ==============================================================================
# 3. ESTRATEGIA DE PROMPTS Y CONTEXTO DEL CATÁLOGO
# ==============================================================================


def formatear_catalogo_para_prompt(catalogo: list[dict] = CATALOGO_DESARROLLOS) -> str:
    """Convierte el catálogo estructurado en un bloque de texto claro para el prompt."""
    lineas = []
    for dev in catalogo:
        linea = (
            f"- {dev['nombre']} | Tipo: {dev['tipo']} | Zona: {dev['zona']} | "
            f"Precio: {dev['precio_texto']} | Detalles: {dev['detalles']}"
        )
        lineas.append(linea)
    lineas.append(
        f"Créditos aceptados en general: {', '.join(CREDITOS_ACEPTADOS)}."
    )
    return "\n".join(lineas)


def construir_system_prompt(catalogo: list[dict] = CATALOGO_DESARROLLOS) -> str:
    """
    Construye el system prompt como contrato operativo del asistente SustentIA.
    """
    catalogo_texto = formatear_catalogo_para_prompt(catalogo)
    return (
        "Eres SustentIA, el asistente virtual de la inmobiliaria Sustentia en Morelia, "
        "Michoacán. Atiendes en español con trato amable, claro, profesional y conciso.\n\n"
        "OBJETIVO DE CONVERSACIÓN:\n"
        "1. Saludar cordialmente al prospecto (especialmente si escribe fuera de horario).\n"
        "2. Identificar su necesidad: tipo de propiedad (casa, departamento o lote), "
        "zona de interés y presupuesto aproximado.\n"
        "3. Recomendar opciones reales del catálogo que se ajusten a su búsqueda.\n"
        "4. Responder preguntas frecuentes (por ejemplo, tipos de crédito aceptados).\n"
        "5. Recordar el nombre y el presupuesto del prospecto durante toda la conversación.\n"
        "6. Capturar amablemente su nombre y teléfono para dar seguimiento.\n"
        "7. Proponer agendar una visita al desarrollo recomendado y mencionar como contacto "
        f"de seguimiento {CONTACTO_ASESOR}.\n\n"
        "CATÁLOGO OFICIAL DE SUSTENTIA (ÚNICA FUENTE DE VERDAD):\n"
        f"{catalogo_texto}\n\n"
        "REGLAS ESTRICTAS (NO ROMPER BAJO NINGUNA CIRCUNSTANCIA):\n"
        "- No inventes desarrollos, precios, promociones, unidades disponibles ni cantidades exactas.\n"
        "- No proporciones direcciones exactas; menciona únicamente la zona general indicada en el catálogo.\n"
        "- Si necesitas referir el teléfono de un asesor, escribe exactamente "
        f"{CONTACTO_ASESOR} y nunca inventes números telefónicos.\n"
        "- Si el prospecto solicita hablar con una persona o plantea un tema legal o fiscal "
        "(impuestos, ISR, escrituración, notaría, contratos legales, demandas, etc.), "
        f"responde EXACTAMENTE: «{MENSAJE_ESCALAMIENTO}».\n"
    )


SYSTEM_PROMPT = construir_system_prompt()


# ==============================================================================
# 4. MEMORIA DE CONVERSACIÓN
# ==============================================================================


def crear_historial(system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Inicializa el historial de mensajes colocando el system prompt al inicio."""
    return [{"role": "system", "content": system_prompt}]


def agregar_mensaje(
    historial: list[dict[str, str]], rol: str, contenido: str
) -> list[dict[str, str]]:
    """Agrega un turno (`user` o `assistant`) al historial existente."""
    historial.append({"role": rol, "content": contenido})
    return historial


def recortar_historial(
    historial: list[dict[str, str]], max_turnos: int = 12
) -> list[dict[str, str]]:
    """
    Conserva el mensaje de sistema en la posición 0 y los últimos `max_turnos`
    intercambios (usuario + asistente) para no exceder la ventana de contexto.
    """
    if not historial:
        return []
    sistema = historial[0]
    mensajes_conversacion = historial[1:]
    max_mensajes = max_turnos * 2
    if len(mensajes_conversacion) <= max_mensajes:
        return historial
    return [sistema] + mensajes_conversacion[-max_mensajes:]


# ==============================================================================
# 5. CAPTURA DE DATOS DEL PROSPECTO Y DETECCIÓN DE ESCALAMIENTO
# ==============================================================================


def _normalizar_texto(texto: str) -> str:
    """Convierte a minúsculas y elimina acentos para comparaciones robustas."""
    nfd = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def crear_datos_prospecto() -> dict:
    """Crea el diccionario inicial donde se guardan los datos del prospecto."""
    return {
        "nombre": None,
        "telefono": None,
        "tipo_propiedad": None,
        "zona": None,
        "presupuesto": None,
        "credito": None,
        "desarrollo_interes": None,
        "escalado": False,
        "proveedor_usado": None,
        "proveedores_por_turno": [],
    }


def _parsear_presupuesto_numerico(presupuesto_texto: str | None) -> int | None:
    """Convierte un texto como '$1,300,000' a entero 1300000."""
    if not presupuesto_texto:
        return None
    digitos = re.sub(r"[^\d]", "", presupuesto_texto)
    return int(digitos) if digitos else None


def _extraer_nombre(mensaje: str) -> str | None:
    """Extrae el nombre propio tras frases introductorias evitando falsos positivos."""
    patrones = [
        r"(?:me\s+llamo|mi\s+nombre\s+es)\s+([A-ZÁÉÍÓÚÑa-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑa-záéíóúñ]+){0,3})",
        r"\bsoy\s+([A-ZÁÉÍÓÚÑa-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑa-záéíóúñ]+){0,3})",
    ]
    palabras_corte = {
        "y",
        "mi",
        "teléfono",
        "telefono",
        "número",
        "numero",
        "cel",
        "celular",
        "de",
        "en",
        "un",
        "una",
        "cliente",
        "prospecto",
        "busco",
        "tengo",
        "para",
        "con",
        "el",
        "la",
        "que",
        "interesado",
        "interesada",
        "nuevo",
        "nueva",
        "muy",
    }
    for patron in patrones:
        coincidencia = re.search(patron, mensaje, flags=re.IGNORECASE)
        if not coincidencia:
            continue
        palabras_crudas = coincidencia.group(1).strip().split()
        palabras_nombre = []
        for palabra in palabras_crudas:
            limpia = re.sub(r"[.,;!?]+$", "", palabra)
            if _normalizar_texto(limpia) in palabras_corte:
                break
            palabras_nombre.append(limpia)
        if palabras_nombre:
            return " ".join(p.capitalize() for p in palabras_nombre)
    return None


def _extraer_telefono(mensaje: str) -> str | None:
    """
    Extrae teléfonos de 10 dígitos (con o sin guiones, espacios o +52),
    ignorando montos monetarios con signo `$`.
    """
    sin_montos = re.sub(r"\$\s*[\d,]+(?:\.\d+)?", " ", mensaje)
    patron = r"(?:\+?52[\s.-]*)?(?:\(?\d{2,3}\)?[\s.-]*\d{3,4}[\s.-]*\d{4}|\b\d{10}\b)"
    for coincidencia in re.finditer(patron, sin_montos):
        texto_match = coincidencia.group(0).strip()
        digitos = re.sub(r"\D", "", texto_match)
        if len(digitos) == 12 and digitos.startswith("52"):
            digitos = digitos[2:]
        if len(digitos) == 10:
            return texto_match
    return None


def _extraer_presupuesto(mensaje: str) -> str | None:
    """Extrae presupuestos en pesos (ej. '$1,300,000' o '1.3 millones')."""
    patron_millones = r"\$?\s*(\d+(?:[.,]\d+)?)\s*(?:millones|mill[oó]n|mdp)\b"
    match_millones = re.search(patron_millones, mensaje, flags=re.IGNORECASE)
    if match_millones:
        valor = float(match_millones.group(1).replace(",", "."))
        return f"${int(round(valor * 1_000_000)):,}"

    patron_pesos = (
        r"(?:\$\s*|presupuesto(?:\s+de)?\s+\$?\s*)(\d{1,3}(?:,\d{3})+|\d{6,8})(?:\.\d+)?"
    )
    match_pesos = re.search(patron_pesos, mensaje, flags=re.IGNORECASE)
    if match_pesos:
        monto = int(match_pesos.group(1).replace(",", ""))
        return f"${monto:,}"

    return None


def capturar_datos_prospecto(mensaje: str, datos_prospecto: dict) -> dict:
    """
    Analiza el mensaje del prospecto y actualiza en sitio el diccionario
    `datos_prospecto` con nombre, teléfono, necesidad y preferencia de crédito.
    """
    normalizado = _normalizar_texto(mensaje)

    nombre = _extraer_nombre(mensaje)
    if nombre:
        datos_prospecto["nombre"] = nombre

    telefono = _extraer_telefono(mensaje)
    if telefono:
        datos_prospecto["telefono"] = telefono

    presupuesto = _extraer_presupuesto(mensaje)
    if presupuesto:
        datos_prospecto["presupuesto"] = presupuesto

    # Identificación de tipo de propiedad
    if re.search(r"\bdepartamentos?\b|\bdepa\b", normalizado):
        datos_prospecto["tipo_propiedad"] = "departamentos"
    elif re.search(r"\blotes?\b|\bterrenos?\b", normalizado):
        datos_prospecto["tipo_propiedad"] = "lotes"
    elif re.search(r"\bcasas?\b", normalizado):
        datos_prospecto["tipo_propiedad"] = "casas"

    # Identificación de zona de interés
    zonas_conocidas = [
        ("walmart la huerta", "Morelia, zona Walmart La Huerta"),
        ("la huerta", "Morelia, zona Walmart La Huerta"),
        ("tecnologico", "Morelia, por el Tecnológico de Morelia"),
        ("la calera", "Morelia, zona La Calera"),
        ("altozano", "Morelia, zona Altozano"),
        ("el vergel", "Morelia, zona El Vergel"),
        ("acuitzio", "Acuitzio del Canje"),
        ("morelia", "Morelia"),
    ]
    for clave_zona, nombre_zona in zonas_conocidas:
        if clave_zona in normalizado:
            datos_prospecto["zona"] = nombre_zona
            break

    # Identificación de crédito
    if "infonavit" in normalizado:
        datos_prospecto["credito"] = "Infonavit"
    elif "fovissste" in normalizado:
        datos_prospecto["credito"] = "Fovissste"
    elif re.search(r"\bbancario\b|\bhipotecario\b|\bbanco\b", normalizado):
        datos_prospecto["credito"] = "Bancario"

    # Identificación explícita de desarrollo
    desarrollos_clave = [
        ("san pedro", "San Pedro"),
        ("torres zojo", "Torres Zojo"),
        ("veronesa", "Veronesa"),
        ("paseo de andalucia", "Paseo de Andalucía (modelos Alba y Leira)"),
        ("andalucia", "Paseo de Andalucía (modelos Alba y Leira)"),
        ("alba", "Paseo de Andalucía (modelos Alba y Leira)"),
        ("leira", "Paseo de Andalucía (modelos Alba y Leira)"),
        ("rincon de las aves", "Rincón de las Aves"),
        ("tepatzicuaro", "Tepatzícuaro"),
        ("valle verde", "Valle Verde"),
    ]
    for clave_dev, nombre_dev in desarrollos_clave:
        if clave_dev in normalizado:
            datos_prospecto["desarrollo_interes"] = nombre_dev
            break

    return datos_prospecto


def detectar_escalamiento(mensaje: str) -> bool:
    """
    Detecta si el prospecto solicita hablar con una persona/asesor humano
    o si pregunta por temas legales o fiscales que requieren asesoría especializada.
    """
    normalizado = _normalizar_texto(mensaje)

    frases_humano = [
        "hablar con un asesor",
        "hablar con una persona",
        "hablar con alguien",
        "hablar con un humano",
        "asesor humano",
        "persona real",
        "agente humano",
        "comunicarme con un asesor",
        "comunicas con un asesor",
        "comuniqueme con un asesor",
        "pasarme con un asesor",
        "pasame con un asesor",
        "quiero un asesor",
        "necesito un asesor",
        "atencion humana",
        "conectar con un asesor",
    ]
    if any(frase in normalizado for frase in frases_humano):
        return True

    terminos_legales_fiscales = [
        r"\blegal(?:es)?\b",
        r"\bfiscal(?:es)?\b",
        r"\bimpuestos?\b",
        r"\bisr\b",
        r"\biva\b",
        r"\bsat\b",
        r"\bnotari[oa]\b",
        r"\bnotarial\b",
        r"\bescrituras?\b",
        r"\bescrituracion\b",
        r"\babogad[oa]s?\b",
        r"\bdemandas?\b",
        r"\blitigios?\b",
        r"\bgravamen(?:es)?\b",
        r"\bfideicomisos?\b",
        r"\bdeducci[oó]n(?:es)?\b",
        r"\bherencias?\b",
        r"\bregimen fiscal\b",
        r"\bconstancia fiscal\b",
    ]
    return any(re.search(patron, normalizado) for patron in terminos_legales_fiscales)


# ==============================================================================
# 6. MODO DE RESPALDO [SIMULADO] Y FUNCIÓN PRINCIPAL DE CONVERSACIÓN
# ==============================================================================


def _filtrar_catalogo(datos_prospecto: dict) -> list[dict]:
    """Filtra opciones del catálogo según tipo de propiedad y presupuesto."""
    tipo = datos_prospecto.get("tipo_propiedad")
    presupuesto_num = _parsear_presupuesto_numerico(datos_prospecto.get("presupuesto"))

    candidatos = []
    for dev in CATALOGO_DESARROLLOS:
        coincide_tipo = True
        if tipo == "casas":
            coincide_tipo = "casa" in dev["tipo"]
        elif tipo == "departamentos":
            coincide_tipo = "departamento" in dev["tipo"]
        elif tipo == "lotes":
            coincide_tipo = "lote" in dev["tipo"]

        coincide_presupuesto = True
        if presupuesto_num is not None and dev["precio_desde"] is not None:
            # Margen de tolerancia del 10% sobre el presupuesto indicado
            coincide_presupuesto = dev["precio_desde"] <= int(presupuesto_num * 1.10)

        if coincide_tipo and coincide_presupuesto:
            candidatos.append(dev)

    if candidatos:
        return candidatos

    # Si no hubo coincidencia exacta en ambos filtros, relajar solo tipo o presupuesto
    if presupuesto_num is not None:
        por_presupuesto = [
            d
            for d in CATALOGO_DESARROLLOS
            if d["precio_desde"] is not None
            and d["precio_desde"] <= int(presupuesto_num * 1.10)
        ]
        if por_presupuesto:
            return por_presupuesto

    return CATALOGO_DESARROLLOS[:2]


def responder_simulado(
    mensaje: str, historial: list[dict[str, str]], datos_prospecto: dict
) -> str:
    """
    Último recurso local: genera una respuesta coherente con el catálogo y la
    memoria de conversación cuando no hay llaves de OpenRouter ni de Groq o
    ambos proveedores fallan. Todas las respuestas se marcan con `[SIMULADO]`.
    """
    normalizado = _normalizar_texto(mensaje)
    nombre = datos_prospecto.get("nombre")
    telefono = datos_prospecto.get("telefono")
    presupuesto = datos_prospecto.get("presupuesto")
    tipo = datos_prospecto.get("tipo_propiedad")
    credito = datos_prospecto.get("credito")
    desarrollo = datos_prospecto.get("desarrollo_interes")

    # Verificar si ya se recomendó o confirmó visita anteriormente en el historial
    historial_asistente = " ".join(
        _normalizar_texto(m["content"])
        for m in historial[:-1]
        if m.get("role") == "assistant"
    )
    menciono_datos_en_turno = bool(
        _extraer_nombre(mensaje) or _extraer_telefono(mensaje)
    )

    # 1. Confirmación de memoria explícita o cierre de horario de visita
    pide_memoria = any(
        frase in normalizado
        for frase in ("recuerdas", "te acuerdas", "olvidaste", "sabes mi")
    )
    propone_horario = bool(
        (nombre or telefono)
        and re.search(
            r"\b(lunes|martes|miercoles|jueves|viernes|sabado|domingo|manana|tarde|horario|am|pm|\d{1,2}:\d{2})\b",
            normalizado,
        )
    )
    if (pide_memoria or propone_horario) and (nombre or presupuesto):
        partes_memoria = []
        if nombre:
            partes_memoria.append(f"claro que sí, {nombre}")
        if presupuesto:
            partes_memoria.append(f"tengo presente tu presupuesto de {presupuesto}")
        if desarrollo:
            partes_memoria.append(f"tu interés en {desarrollo}")

        if len(partes_memoria) > 1:
            resumen_memoria = f"{', '.join(partes_memoria[:-1])} y {partes_memoria[-1]}"
        else:
            resumen_memoria = partes_memoria[0]

        respuesta = f"[SIMULADO] ¡Por supuesto, {resumen_memoria}! "
        if propone_horario:
            respuesta += (
                "Tomo nota de tu preferencia de horario para la visita. "
                f"Un asesor de Sustentia se comunicará contigo desde {CONTACTO_ASESOR} "
                "para confirmar la cita y compartirte el punto de encuentro en la zona."
            )
        else:
            respuesta += (
                "¿Te gustaría que agendemos tu visita? Un asesor te confirmará "
                f"a través de {CONTACTO_ASESOR}."
            )
        return respuesta

    # 2. Captura reciente de nombre y/o teléfono
    if menciono_datos_en_turno:
        if nombre and telefono:
            contexto_busqueda = ""
            if desarrollo and presupuesto:
                contexto_busqueda = (
                    f" sobre {desarrollo} considerando tu presupuesto de {presupuesto}"
                )
            elif presupuesto:
                contexto_busqueda = f" considerando tu presupuesto de {presupuesto}"
            return (
                f"[SIMULADO] ¡Muchas gracias, {nombre}! Ya registré tu teléfono "
                f"({telefono}) para dar seguimiento{contexto_busqueda}. "
                "¿Qué día y horario te quedaría mejor para agendar una visita y conocer el desarrollo? "
                f"Nuestro equipo confirmará tu cita por medio de {CONTACTO_ASESOR}."
            )
        if nombre and not telefono:
            return (
                f"[SIMULADO] ¡Mucho gusto, {nombre}! Para poder enviarte la información "
                "y coordinar una visita, ¿me compartirías también un número de teléfono?"
            )
        if telefono and not nombre:
            return (
                f"[SIMULADO] ¡Gracias! Ya anoté el teléfono {telefono}. "
                "¿Cuál es tu nombre para registrarte y proponerte horarios de visita?"
            )

    # 3. Pregunta sobre créditos aceptados
    if credito or any(
        p in normalizado for p in ("credito", "financiamiento", "infonavit", "fovissste")
    ):
        credito_txt = credito if credito else "Infonavit, Fovissste y bancario"
        ref_dev = f" en {desarrollo}" if desarrollo else ""
        ref_presupuesto = (
            f" dentro de tu presupuesto de {presupuesto}" if presupuesto else ""
        )
        solicitud_datos = ""
        if not nombre or not telefono:
            solicitud_datos = (
                " Para enviarte más detalles y revisar opciones de visita, "
                "¿me compartes por favor tu nombre y número de teléfono?"
            )
        else:
            solicitud_datos = (
                f" {nombre}, ¿te gustaría que propongamos fecha para una visita? "
                f"Puedes esperar confirmación al {CONTACTO_ASESOR}."
            )
        return (
            f"[SIMULADO] ¡Sí! En Sustentia aceptamos créditos Infonavit, Fovissste y bancario, "
            f"por lo que puedes utilizar tu crédito {credito_txt}{ref_dev}{ref_presupuesto}. "
            "La disponibilidad vigente y requisitos particulares los revisa directamente un asesor "
            "para darte información actualizada."
            f"{solicitud_datos}"
        )

    # 4. Búsqueda con criterios (tipo, zona o presupuesto)
    if tipo or presupuesto or datos_prospecto.get("zona"):
        opciones = _filtrar_catalogo(datos_prospecto)
        principal = opciones[0]
        if not datos_prospecto.get("desarrollo_interes"):
            datos_prospecto["desarrollo_interes"] = principal["nombre"]

        if nombre:
            inicio_rec = f"¡Excelente elección, {nombre}! Con tu presupuesto de {presupuesto}" if presupuesto else f"¡Excelente elección, {nombre}! Para lo que buscas"
        else:
            inicio_rec = f"¡Excelente elección! Con tu presupuesto de {presupuesto}" if presupuesto else "¡Excelente elección! Para lo que buscas"
        recomendacion = (
            f"[SIMULADO] {inicio_rec}, nuestra mejor opción es **{principal['nombre']}** "
            f"({principal['tipo']} en {principal['zona']}, {principal['precio_texto']}, "
            f"{principal['detalles']})."
        )

        # Si hay una segunda opción cercana (por ejemplo Torres Zojo), mencionarla brevemente
        alternativas = [
            d
            for d in CATALOGO_DESARROLLOS
            if d["nombre"] != principal["nombre"]
            and d["precio_desde"] is not None
            and _parsear_presupuesto_numerico(presupuesto) is not None
            and d["precio_desde"] <= int(_parsear_presupuesto_numerico(presupuesto) * 1.10)
        ]
        if alternativas:
            alt = alternativas[0]
            recomendacion += (
                f" Si también consideraras {alt['tipo']}, contamos con **{alt['nombre']}** "
                f"({alt['zona']}, {alt['precio_texto']})."
            )

        if "agendar visita" not in historial_asistente:
            recomendacion += (
                " ¿Te gustaría saber qué créditos aceptamos (Infonavit, Fovissste o bancario) "
                "o prefieres compartir tu nombre y teléfono para agendar una visita?"
            )
        return recomendacion

    # 5. Saludo inicial sin criterios todavía
    return (
        "[SIMULADO] ¡Hola! Bienvenido a Sustentia, tu inmobiliaria en Morelia. "
        "Con gusto te atiendo en este momento, incluso fuera de horario. "
        "Para recomendarte las mejores opciones de nuestro catálogo, cuéntame por favor: "
        "¿buscas casa, departamento o lote, en qué zona y cuál es tu presupuesto aproximado?"
    )


def _registrar_proveedor(datos: dict, proveedor: str) -> None:
    """Registra el proveedor que respondió el turno actual y el historial por turno."""
    datos["proveedor_usado"] = proveedor
    datos.setdefault("proveedores_por_turno", []).append(proveedor)


def _llamar_proveedor_openai(
    cliente_api,
    modelo: str,
    mensajes: list[dict[str, str]],
) -> str:
    """Invoca un endpoint compatible con OpenAI y valida que el contenido no esté vacío."""
    respuesta_api = cliente_api.chat.completions.create(
        model=modelo,
        messages=mensajes,
        temperature=0.3,
        max_tokens=450,
    )
    contenido = (respuesta_api.choices[0].message.content or "").strip()
    if not contenido:
        raise ValueError(f"El modelo {modelo} devolvió una respuesta vacía.")
    return contenido


def conversar(
    mensaje: str,
    historial: list[dict[str, str]],
    datos_prospecto: dict | None = None,
    cliente=None,
    modelo: str = MODELO_LLAMA,
    forzar_simulado: bool = False,
    cliente_respaldo=None,
    modelo_respaldo: str = MODELO_RESPALDO_GROQ,
) -> str:
    """
    Función principal del asistente con cadena de continuidad en 3 niveles:
    1. Captura datos del prospecto y registra el mensaje del usuario en el historial.
    2. Evalúa reglas de escalamiento deterministas (solicitud de persona o tema legal/fiscal).
    3. Intenta responder con el proveedor principal (OpenRouter / Llama).
    4. Ante cualquier error (402, 401, 429, red o servidor) o ausencia de llave
       principal, reintenta el mismo turno con Groq (`openai/gpt-oss-120b`, que NO es
       Llama sino respaldo de continuidad) preservando intacto el historial.
    5. Si Groq tampoco está disponible o falla, recurre a `[SIMULADO]`.
    """
    datos = datos_prospecto if datos_prospecto is not None else DATOS_PROSPECTO_GLOBAL
    cliente_principal = cliente if cliente is not None else CLIENTE_OPENROUTER_GLOBAL
    cliente_groq_activo = (
        cliente_respaldo if cliente_respaldo is not None else CLIENTE_GROQ_GLOBAL
    )

    capturar_datos_prospecto(mensaje, datos)
    agregar_mensaje(historial, "user", mensaje)

    # Regla de escalamiento prioritaria (defensa en profundidad en código)
    if detectar_escalamiento(mensaje):
        datos["escalado"] = True
        _registrar_proveedor(datos, "Escalamiento local (sin LLM)")
        agregar_mensaje(historial, "assistant", MENSAJE_ESCALAMIENTO)
        return MENSAJE_ESCALAMIENTO

    if not forzar_simulado:
        mensajes_envio = recortar_historial(historial)

        if cliente_principal is not None:
            try:
                contenido = _llamar_proveedor_openai(
                    cliente_principal, modelo, mensajes_envio
                )
                _registrar_proveedor(datos, f"OpenRouter ({modelo})")
                agregar_mensaje(historial, "assistant", contenido)
                return contenido
            except Exception as exc:
                if cliente_groq_activo is not None:
                    print(
                        f"Aviso: error en OpenRouter ({exc}). "
                        f"Reintentando turno con respaldo Groq ({modelo_respaldo})."
                    )
                else:
                    print(
                        f"Aviso: error en OpenRouter ({exc}). "
                        "Sin respaldo Groq disponible; activando modo [SIMULADO]."
                    )

        if cliente_groq_activo is not None:
            try:
                contenido = _llamar_proveedor_openai(
                    cliente_groq_activo, modelo_respaldo, mensajes_envio
                )
                _registrar_proveedor(datos, f"Groq ({modelo_respaldo})")
                agregar_mensaje(historial, "assistant", contenido)
                return contenido
            except Exception as exc:
                print(
                    f"Aviso: error en respaldo Groq ({exc}). "
                    "Activando último recurso [SIMULADO]."
                )

    respuesta_local = responder_simulado(mensaje, historial, datos)
    _registrar_proveedor(datos, "[SIMULADO]")
    agregar_mensaje(historial, "assistant", respuesta_local)
    return respuesta_local


# Estado global por defecto para invocaciones directas `conversar(mensaje, historial)`
DATOS_PROSPECTO_GLOBAL = crear_datos_prospecto()
CLIENTE_OPENROUTER_GLOBAL = None
CLIENTE_GROQ_GLOBAL = None


# ==============================================================================
# 7. EJECUCIÓN DE DEMOSTRACIÓN
# ==============================================================================


def ejecutar_demo(forzar_simulado: bool = False) -> None:
    """
    Ejecuta la conversación de ejemplo completa:
    saludo -> búsqueda de casa ($1,300,000) -> recomendación de San Pedro ->
    duda de Infonavit -> captura de nombre y teléfono ficticio -> propuesta de visita
    + verificación de memoria y pruebas de escalamiento.
    """
    global CLIENTE_OPENROUTER_GLOBAL, CLIENTE_GROQ_GLOBAL

    api_key_openrouter = "" if forzar_simulado else obtener_api_key(permitir_prompt=False)
    api_key_groq = "" if forzar_simulado else obtener_api_key_groq(permitir_prompt=False)
    cliente = None if forzar_simulado else crear_cliente_openrouter(api_key_openrouter)
    cliente_respaldo = None if forzar_simulado else crear_cliente_groq(api_key_groq)
    CLIENTE_OPENROUTER_GLOBAL = cliente
    CLIENTE_GROQ_GLOBAL = cliente_respaldo
    proveedor_activo = describir_proveedor_activo(
        cliente, cliente_respaldo, forzar_simulado=forzar_simulado
    )

    print("=" * 72)
    print("SustentIA — Asistente Inteligente con Llama")
    print(f"Proveedor activo: {proveedor_activo}")
    print(f"Principal: OpenRouter ({OPENROUTER_BASE_URL}) | Modelo: {MODELO_LLAMA}")
    print(
        f"Respaldo automático: Groq ({GROQ_BASE_URL}) | Modelo: {MODELO_RESPALDO_GROQ} "
        "(no es Llama; continuidad operativa)"
    )
    print("Último recurso: [SIMULADO]")
    print("=" * 72)

    historial = crear_historial()
    datos = crear_datos_prospecto()

    # Nota: el teléfono 443-000-0000 es un número ficticio de prueba para la demo.
    turnos_demo = [
        "Hola, buenas noches, ¿hay alguien disponible para darme información?",
        "Busco una casa en Morelia, mi presupuesto es de $1,300,000.",
        "¡Qué bien! ¿Aceptan crédito Infonavit?",
        "Me llamo Sofía Ramírez y mi teléfono es 443-000-0000.",
        "Me gustaría agendar visita este sábado a las 11:00 am. Por cierto, ¿recuerdas mi nombre y mi presupuesto?",
    ]

    for indice, mensaje_usuario in enumerate(turnos_demo, start=1):
        print(f"\n[Turno {indice}] Prospecto: {mensaje_usuario}")
        respuesta = conversar(
            mensaje=mensaje_usuario,
            historial=historial,
            datos_prospecto=datos,
            cliente=cliente,
            cliente_respaldo=cliente_respaldo,
            forzar_simulado=forzar_simulado,
        )
        print(
            f"[Turno {indice} | Proveedor: {datos['proveedor_usado']}] SustentIA: {respuesta}"
        )

    print("\n" + "-" * 72)
    print("Datos capturados del prospecto en memoria:")
    for clave, valor in datos.items():
        print(f"  - {clave}: {valor}")
    print(f"Total de mensajes en historial (incluyendo system): {len(historial)}")

    print("\n" + "-" * 72)
    print("Pruebas de detección de escalamiento:")
    pruebas_escalamiento = [
        ("Tema fiscal/legal", "¿Cómo se calcula el ISR y cuánto cuestan las escrituras con notario?"),
        ("Solicitud de persona", "Prefiero hablar con una persona, ¿me comunicas con un asesor?"),
    ]
    for etiqueta, msg_esc in pruebas_escalamiento:
        historial_esc = crear_historial()
        datos_esc = crear_datos_prospecto()
        resp_esc = conversar(
            mensaje=msg_esc,
            historial=historial_esc,
            datos_prospecto=datos_esc,
            cliente=cliente,
            cliente_respaldo=cliente_respaldo,
            forzar_simulado=forzar_simulado,
        )
        print(f"  [{etiqueta}] Prospecto: {msg_esc}")
        print(
            f"  [{etiqueta}] SustentIA: {resp_esc} "
            f"(escalado={datos_esc['escalado']}, proveedor={datos_esc['proveedor_usado']})"
        )
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ejecuta la demo del asistente inmobiliario SustentIA."
    )
    parser.add_argument(
        "--simulado",
        action="store_true",
        help="Fuerza la ejecución en modo [SIMULADO] sin usar OpenRouter ni Groq.",
    )
    argumentos = parser.parse_args()
    ejecutar_demo(forzar_simulado=argumentos.simulado)
