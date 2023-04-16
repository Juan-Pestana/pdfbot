import type { NextApiRequest, NextApiResponse } from 'next';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { PineconeStore } from 'langchain/vectorstores';
import { makeChain } from '@/utils/makechain';
import { pinecone } from '@/utils/pinecone-client';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';
import withAllowCORS from '@/middleware/withAllowCORS';

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  const { question, history } = req.body;

  if (!question) {
    return res.status(400).json({ message: 'No question in the request' });
  }
  // OpenAI recommends replacing newlines with spaces for best results
  const sanitizedQuestion = question.trim().replaceAll('\n', ' ');

  const index = pinecone.Index(PINECONE_INDEX_NAME);

  /* create vectorstore*/
  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings({}),
    {
      pineconeIndex: index,
      textKey: 'text',
      namespace: PINECONE_NAME_SPACE,
    },
  );

  //   res.writeHead(200, {
  //     'Content-Type': 'application/json',
  //     'Cache-Control': 'no-cache, no-transform',
  //     Connection: 'keep-alive',
  //   });

  const sendData = (data: string) => {
    res.write(`data: ${data}\n\n`);
  };

  //sendData(JSON.stringify({ data: '' }));

  //create chain
  const chain = makeChain(vectorStore, (token: string) => {
    sendData(JSON.stringify({ data: token }));
  });

  try {
    //Ask a question
    const response = await chain.call({
      question: sanitizedQuestion,
      chat_history: history || [],
    });
    //console.log('esta es la respuesta', response);
    //console.log('response', response);
    // sendData(JSON.stringify({ sourceDocs: response.sourceDocuments }));
    return res.status(200).json(response);
  } catch (error) {
    console.log('error', error);
  }
};

export default withAllowCORS(handler);
