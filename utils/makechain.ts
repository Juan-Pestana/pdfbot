import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `Eres un asistente de IA que brinda consejos útiles. Se le dan las siguientes partes extraídas de un documento extenso y una pregunta. Proporcione una respuesta conversacional basada en el contexto proporcionado.
  Solo debe proporcionar hipervínculos que hagan referencia al contexto a continuación. NO inventes hipervínculos.
  Si no puede encontrar la respuesta en el contexto a continuación, simplemente diga "Hmm, no estoy seguro". No intentes inventar una respuesta.
  Si la pregunta no está relacionada con el contexto, responda cortésmente que está sintonizado para responder solo preguntas relacionadas con el contexto.

Question: {question}
=========
{context}
=========
Responde en Markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0,
      modelName: 'gpt-3.5-turbo', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: false,
      //streaming: Boolean(onTokenStream),
      // callbackManager: onTokenStream
      //   ? CallbackManager.fromHandlers({
      //       async handleLLMNewToken(token) {
      //         onTokenStream(token);
      //         // console.log(token);
      //       },
      //     })
      //   : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: false,
    k: 1, //number of source documents to return
  });
};
