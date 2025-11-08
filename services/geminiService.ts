import { GoogleGenAI, Type } from "@google/genai";
import { DataRow, DescriptiveStats } from '../types';

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  // In a real app, you'd handle this more gracefully.
  // Here we assume it's set in the environment.
  console.warn("API_KEY for Gemini is not set in environment variables.");
}

const ai = new GoogleGenAI({ apiKey: API_KEY! });

export const getDatasetInsights = async (
  headers: string[],
  stats: DescriptiveStats,
  sampleData: DataRow[]
): Promise<string> => {
  if (!API_KEY) {
    return "Error: La clave API de Gemini no está configurada. Por favor, configúrela en las variables de entorno.";
  }

  const model = 'gemini-2.5-flash';
  
  const statsString = JSON.stringify(stats, null, 2);
  const sampleDataString = JSON.stringify(sampleData, null, 2);

  const prompt = `
    Eres un analista de datos experto. Tu tarea es proporcionar un análisis conciso y útil de un conjunto de datos para la predicción de ventas.

    Aquí está la información del conjunto de datos:
    1.  **Columnas**: ${headers.join(', ')}
    2.  **Estadísticas Descriptivas (para columnas numéricas)**:
        \`\`\`json
        ${statsString}
        \`\`\`
    3.  **Primeras 5 filas de datos**:
        \`\`\`json
        ${sampleDataString}
        \`\`\`

    Por favor, proporciona un análisis que incluya:
    -   Una breve descripción general de los datos.
    -   Posibles relaciones o patrones interesantes que observes entre las variables.
    -   Recomendaciones sobre qué variables podrían ser buenos predictores para un modelo de regresión.
    -   Advertencias sobre posibles problemas como multicolinealidad, valores atípicos (outliers) o la necesidad de transformar alguna variable.

    Formatea tu respuesta en Markdown para una fácil lectura. Sé claro, conciso y orientado a la acción.
    `;

  try {
    const response = await ai.models.generateContent({
        model: model,
        contents: prompt
    });
    return response.text;
  } catch (error) {
    console.error("Error calling Gemini API:", error);
    return `Error al contactar la API de Gemini. Por favor, revisa la consola para más detalles. Detalles: ${error instanceof Error ? error.message : String(error)}`;
  }
};

export const getVariableSuggestions = async (
  headers: string[],
  sampleData: DataRow[]
): Promise<{ dependentVar: string; independentVars: string[] }> => {
  if (!API_KEY) {
    throw new Error("La clave API de Gemini no está configurada.");
  }

  const model = 'gemini-2.5-flash';
  const sampleDataString = JSON.stringify(sampleData.slice(0, 5), null, 2);

  const prompt = `
    Analiza los siguientes encabezados de columnas y datos de muestra de un conjunto de datos.
    Tu tarea es actuar como un científico de datos y sugerir la mejor variable dependiente (objetivo) y las mejores variables independientes (predictores) para construir un modelo de predicción de ventas.

    La variable dependiente debe ser la que probablemente represente las ventas (ej. 'ventas', 'ingresos', 'unidades_vendidas').
    Las variables independientes deben ser aquellas que lógicamente podrían influir en la variable de ventas.

    Encabezados: ${headers.join(', ')}

    Primeras 5 filas de datos de muestra:
    \`\`\`json
    ${sampleDataString}
    \`\`\`

    Devuelve tu sugerencia únicamente en formato JSON.
    `;
    
  try {
    const response = await ai.models.generateContent({
      model: model,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            dependentVar: {
              type: Type.STRING,
              description: 'El nombre de la columna que mejor representa la variable dependiente (ej. ventas).',
            },
            independentVars: {
              type: Type.ARRAY,
              items: {
                type: Type.STRING,
              },
              description: 'Una lista de nombres de columnas que son buenos predictores para la variable dependiente.',
            },
          },
          required: ['dependentVar', 'independentVars'],
        },
      },
    });

    const jsonText = response.text.trim();
    const suggestions = JSON.parse(jsonText);
    
    if (!suggestions.dependentVar || !Array.isArray(suggestions.independentVars)) {
        throw new Error('La respuesta de la IA no tiene el formato esperado.');
    }

    return suggestions;
    
  } catch (error) {
    console.error("Error calling Gemini API for suggestions:", error);
    throw new Error(`Error al obtener sugerencias de la IA. Detalles: ${error instanceof Error ? error.message : String(error)}`);
  }
};
