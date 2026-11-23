"""
SustentIA — Asistente Inteligente con Llama (Hackathon 1).

Asistente conversacional para la inmobiliaria Sustentia (Morelia, Michoacán) con
cadena de continuidad `OpenRouter (Llama) → Groq (respaldo) → mensaje de espera`
sobre la interfaz compatible con OpenAI:
1. Proveedor principal: Meta Llama (`meta-llama/llama-3.1-8b-instruct`, con
   alternativa `meta-llama/llama-3.3-70b-instruct` de mayor fidelidad al prompt)
   vía OpenRouter (`https://openrouter.ai/api/v1`).
2. Respaldo automático: Groq (`https://api.groq.com/openai/v1`) con el modelo
   `openai/gpt-oss-120b`. Nota importante de transparencia: `openai/gpt-oss-120b`
   NO es un modelo Llama; su función es mantener la continuidad operativa si
   OpenRouter falla o agota sus créditos.
3. Sin modelo disponible (mensaje de espera): si no hay llaves configuradas o
   si ambos proveedores fallan, el asistente devuelve `MENSAJE_SIN_MODELO`,
   marca `escalado=True` y registra el proveedor del turno como
   `Sin modelo disponible — mensaje de espera enviado` para que un asesor
   atienda al prospecto.

Atiende prospectos fuera de horario, identifica necesidad y presupuesto,
recomienda desarrollos del catálogo sin inventar datos, recuerda la conversación,
captura datos del prospecto y escala a un asesor humano cuando corresponde.
"""

from __future__ import annotations

import getpass
import os
import re
import sys
import unicodedata

# Importación opcional del SDK de OpenAI (compatible con OpenRouter y Groq):
# permite importar el módulo aunque `openai` no esté instalado; sin paquete o
# sin llaves disponibles, `conversar()` devuelve `MENSAJE_SIN_MODELO` y marca
# la conversación como escalada para seguimiento de un asesor.
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
MENSAJE_SIN_MODELO = "En este momento no podemos atenderte, pero a primera hora un asesor se comunicará contigo."

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
) -> str:
    """Describe de forma clara la disponibilidad de OpenRouter y Groq."""
    if cliente_openrouter is None and cliente_groq is None:
        return "Sin proveedores disponibles (OpenRouter y Groq sin llaves configuradas)"
    if cliente_openrouter is not None and cliente_groq is not None:
        return (
            f"OpenRouter ({MODELO_LLAMA}) + respaldo automático Groq ({MODELO_RESPALDO_GROQ})"
        )
    if cliente_openrouter is not None:
        return f"OpenRouter ({MODELO_LLAMA}) [sin respaldo Groq]"
    return f"Groq ({MODELO_RESPALDO_GROQ}) [OpenRouter no configurado]"


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
# 6. FUNCIÓN PRINCIPAL DE CONVERSACIÓN Y CADENA DE CONTINUIDAD
# ==============================================================================


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
    cliente_respaldo=None,
    modelo_respaldo: str = MODELO_RESPALDO_GROQ,
) -> str:
    """
    Función principal del asistente con cadena de continuidad:
    `OpenRouter (Llama) → Groq (respaldo) → mensaje de espera`.

    1. Captura datos del prospecto y registra el mensaje del usuario en el historial.
    2. Evalúa reglas de escalamiento deterministas (solicitud de persona o tema legal/fiscal).
    3. Intenta responder con el proveedor principal (OpenRouter / Llama).
    4. Ante cualquier error (402, 401, 429, red o servidor) o ausencia de llave
       principal, reintenta el mismo turno con Groq (`openai/gpt-oss-120b`, que NO es
       Llama sino respaldo de continuidad) preservando intacto el historial.
    5. Si Groq tampoco está disponible o también falla, marca `escalado=True`,
       registra `Sin modelo disponible — mensaje de espera enviado` y devuelve `MENSAJE_SIN_MODELO`.
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
                    "Sin respaldo Groq disponible; enviando mensaje de espera y escalando a un asesor."
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
                "Sin modelos disponibles; enviando mensaje de espera y escalando a un asesor."
            )

    datos["escalado"] = True
    _registrar_proveedor(datos, "Sin modelo disponible — mensaje de espera enviado")
    agregar_mensaje(historial, "assistant", MENSAJE_SIN_MODELO)
    return MENSAJE_SIN_MODELO


# Estado global por defecto para invocaciones directas `conversar(mensaje, historial)`
DATOS_PROSPECTO_GLOBAL = crear_datos_prospecto()
CLIENTE_OPENROUTER_GLOBAL = None
CLIENTE_GROQ_GLOBAL = None


# ==============================================================================
# 7. EJECUCIÓN DE DEMOSTRACIÓN
# ==============================================================================


def ejecutar_demo() -> None:
    """
    Ejecuta la conversación de ejemplo completa cuando existe al menos un
    proveedor configurado (`OPENROUTER_API_KEY` o `GROQ_API_KEY`):
    saludo -> búsqueda de casa ($1,300,000) -> recomendación de San Pedro ->
    duda de Infonavit -> captura de nombre y teléfono ficticio -> propuesta de visita
    + verificación de memoria y pruebas de escalamiento.
    Si no hay llaves configuradas, muestra un aviso y termina sin fingir una demo.
    """
    global CLIENTE_OPENROUTER_GLOBAL, CLIENTE_GROQ_GLOBAL

    api_key_openrouter = obtener_api_key(permitir_prompt=False)
    api_key_groq = obtener_api_key_groq(permitir_prompt=False)
    cliente = crear_cliente_openrouter(api_key_openrouter)
    cliente_respaldo = crear_cliente_groq(api_key_groq)
    CLIENTE_OPENROUTER_GLOBAL = cliente
    CLIENTE_GROQ_GLOBAL = cliente_respaldo
    proveedor_activo = describir_proveedor_activo(cliente, cliente_respaldo)

    print("=" * 72)
    print("SustentIA — Asistente Inteligente con Llama")
    print(f"Proveedor activo: {proveedor_activo}")
    print(f"Principal: OpenRouter ({OPENROUTER_BASE_URL}) | Modelo: {MODELO_LLAMA}")
    print(
        f"Respaldo automático: Groq ({GROQ_BASE_URL}) | Modelo: {MODELO_RESPALDO_GROQ} "
        "(no es Llama; continuidad operativa)"
    )
    print("Sin modelo disponible: mensaje predefinido de espera + escalamiento a asesor")
    print("=" * 72)

    if cliente is None and cliente_respaldo is None:
        if not api_key_openrouter and not api_key_groq:
            print(
                "\nAviso: no hay ninguna API key configurada. "
                "Para ejecutar la demostración necesitas configurar al menos "
                "`OPENROUTER_API_KEY` (proveedor principal Llama) o "
                "`GROQ_API_KEY` (respaldo automático)."
            )
        elif OpenAI is None:
            print(
                "\nAviso: falta instalar el paquete `openai` (`pip install openai`) "
                "para conectarse con `OPENROUTER_API_KEY` o `GROQ_API_KEY`."
            )
        print(
            "En una conversación real sin modelo disponible, el prospecto recibiría el "
            f"mensaje de espera («{MENSAJE_SIN_MODELO}») y la conversación quedaría "
            "marcada como escalada para que un asesor la atienda."
        )
        print("=" * 72)
        return

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
        )
        print(f"  [{etiqueta}] Prospecto: {msg_esc}")
        print(
            f"  [{etiqueta}] SustentIA: {resp_esc} "
            f"(escalado={datos_esc['escalado']}, proveedor={datos_esc['proveedor_usado']})"
        )
    print("=" * 72)


if __name__ == "__main__":
    ejecutar_demo()
