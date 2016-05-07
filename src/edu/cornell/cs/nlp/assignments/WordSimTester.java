package edu.cornell.cs.nlp.assignments;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;
import org.apache.commons.math3.stat.ranking.NaNStrategy;
import org.apache.commons.math3.stat.ranking.NaturalRanking;

import edu.cornell.cs.nlp.assignments.util.CommandLineUtils;
import edu.cornell.cs.nlp.assignments.util.HashId;
import edu.cornell.cs.nlp.assignments.util.Pair;

public class WordSimTester {

	public static void main(String[] args) throws Exception {
		// Parse command line flags and arguments.
		final Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Read commandline parameters.
		String embeddingPath = "";
		if (!argMap.containsKey("-embeddings")) {
			System.out.println("-embeddings flag required.");
			System.exit(0);
		} else {
			embeddingPath = argMap.get("-embeddings");
		}

		String wordSimPath = "";
		if (!argMap.containsKey("-wordsim")) {
			System.out.println("-wordsim flag required.");
			System.exit(0);
		} else {
			wordSimPath = argMap.get("-wordsim");
		}

		// Read in the labeled similarities and generate the target vocabulary.
		System.out.println("Loading wordsim353 ...");
		final List<Pair<Pair<String, String>, Float>> wordSimPairs = readWordSimPairs(
				wordSimPath);
		final Set<String> targetVocab = getWordSimVocab(wordSimPath);

		// It is likely that you will want to generate your embeddings
		// elsewhere. But this supports the option to generate the embeddings
		// and evaluate them in a single loop.
		HashMap<String, float[]> embeddings;
		if (argMap.containsKey("-trainandeval")) {
			// Get some training data.
			String dataPath = "";
			if (!argMap.containsKey("-trainingdata")) {
				System.out.println(
						"-trainingdata flag required with -trainandeval");
				System.exit(0);
			} else {
				dataPath = argMap.get("-trainingdata");
			}

            // Since this simple approach does not do dimensionality reduction
            // on the co-occurrence vectors, we instead control the size of the
            // vectors by only counting co-occurrence with core WordNet senses.
            String wordNetPath = "";
            if (!argMap.containsKey("-wordnetdata")) {
                System.out.println(
                        "-wordnetdata flag required with -trainandeval");
                System.exit(0);
            } else {
                wordNetPath = argMap.get("-wordnetdata");
            }
            final HashMap<String, Integer> contentWordVocab = getWordNetVocab(
                    wordNetPath);
            
            int embeddingSize = 0;
            
            if (argMap.containsKey("-baseline")) {
                System.out.println("Training embeddings on " + dataPath + " with baseline...");
                embeddings = getEmbeddings(dataPath, contentWordVocab, targetVocab);
                embeddingSize = contentWordVocab.size();
            } else if (argMap.containsKey("-baseline1")) {
                System.out.println("Training embeddings on " + dataPath + " with baseline1...");
                embeddings = getEmbeddingsD(dataPath, contentWordVocab, targetVocab, true);
                embeddingSize = contentWordVocab.size();
            } else if (argMap.containsKey("-baseline2")) {
                System.out.println("Training embeddings on " + dataPath + " with baseline2...");
                embeddings = getEmbeddingsD(dataPath, contentWordVocab, targetVocab, false);
                embeddingSize = contentWordVocab.size();
            } else if (argMap.containsKey("-deponly")) {
                System.out.println("Training embeddings on " + dataPath + " with deponly...");
                Pair<HashMap<String, float[]>, Integer> pair = getDepEmbeddings(dataPath, null, targetVocab);
                embeddings = pair.getFirst();
                embeddingSize = pair.getSecond();
            } else if (argMap.containsKey("-dep")) {
                System.out.println("Training embeddings on " + dataPath + " with dep...");
                Pair<HashMap<String, float[]>, Integer> pair = getDepEmbeddings(dataPath, contentWordVocab, targetVocab);
                embeddings = pair.getFirst();
                embeddingSize = pair.getSecond();
            } else {
                System.out.println("Training embeddings on " + dataPath + " with skip-gram...");
                Pair<HashMap<String, float[]>, Integer> pair = getDepEmbeddings(dataPath, contentWordVocab, targetVocab);
                embeddings = pair.getFirst();
                embeddingSize = pair.getSecond();
                
                ObjectiveFunction objfun = new ObjectiveFunction(embeddings);
                double[] initialweights = new double[embeddingSize];
                for (int i = 0; i < embeddingSize; i++) {
                    initialweights[i] = 1.0;
                }
                
                // MAXIMIZE
                double[] weights = initialweights;
                
                for (final String word : embeddings.keySet()) {
                    float[] contexts = embeddings.get(word);
                    for (int i = 0; i < embeddingSize; i++) {
                        contexts[i] = contexts[i] * (float) weights[i];
                    }
                }
            }

			// Keep only the words that are needed.
			System.out.println("Writing embeddings to " + embeddingPath + " ...");
			embeddings = reduceVocabulary(embeddings, targetVocab);
			writeEmbeddings(embeddings, embeddingPath, embeddingSize);
		} else {
			// Read in embeddings.
			System.out.println("Loading embeddings ...");
			embeddings = readEmbeddings(embeddingPath);

			// Keep only the words that are needed.
			System.out.println("Writing reduced vocabbulary embeddings to "
					+ embeddingPath + ".reduced ...");
			embeddings = reduceVocabulary(embeddings, targetVocab);
			writeEmbeddings(embeddings, embeddingPath + ".reduced",
					embeddings.values().iterator().next().length);
		}

		reduceVocabulary(embeddings, targetVocab);

		final double score = spearmansScore(wordSimPairs, embeddings);
		System.out.println("Score is " + score);

	}

	/**
	 * Find the cosine similarity of two embedding vectors. Fail if they have
	 * different dimensionalities.
	 *
	 * @param embedding1
	 * @param embedding2
	 * @return
	 * @throws Exception
	 */
	private static double cosineSimilarity(float[] embedding1,
			float[] embedding2) throws Exception {
		if (embedding1.length != embedding2.length) {
			System.out.println("Embeddings have different dimensionalities: "
					+ embedding1.length + " vs. " + embedding2.length);
			System.exit(0);
		}

		double innerProduct = 0;
		double squaredMagnitude1 = 0;
		double squaredMagnitude2 = 0;
		for (int i = 0; i < embedding1.length; i++) {
			innerProduct += embedding1[i] * embedding2[i];
			squaredMagnitude1 += Math.pow(embedding1[i], 2);
			squaredMagnitude2 += Math.pow(embedding2[i], 2);
		}

		return (float) (innerProduct / (Math.sqrt(squaredMagnitude1)
				* Math.sqrt(squaredMagnitude2)));

	}

	/**
	 * A dumb vector space model that counts each word's co-occurences with a
	 * predefined set of content words and uses these co-occurence vectors
	 * directly as word representations. The context in which a word occurs is
	 * the set of content words in an entire sentence.
	 *
	 * N.B. Most people would probably not consider this an embedding model,
	 * since the words have not been embedded in a lower dimensional subspace.
	 * However, it is a good starting point.
	 *
	 * Since this approach does not share any information between
	 * representations of different words, we can filter the training data to
	 * only include sentences that contain words of interest. In other
	 * approaches this may not be a good idea.
	 *
	 * @param data_path
	 * @param target_vocab
	 * @param embedding_map
	 * @return
	 */
	private static HashMap<String, float[]> getEmbeddings(String dataPath,
			HashMap<String, Integer> contentVocab, Set<String> targetVocab) {

		final HashMap<String, float[]> embeddingMatrix = new HashMap<String, float[]>();
		for (final String target_word : targetVocab) {
			embeddingMatrix.put(target_word, new float[contentVocab.size()]);
		}

		final Collection<List<String>> sentenceCollection = SentenceCollection.Reader
				.readSentenceCollection(dataPath);

		for (final List<String> sentence : sentenceCollection) {
			final Set<String> sw = new HashSet<String>(sentence);
			sw.retainAll(targetVocab);
			for (final String word : sentence) {
				if (!contentVocab.containsKey(word)) {
					continue;
				}
				final int contentWordId = contentVocab.get(word);
				for (final String targetWord : sw) {
					embeddingMatrix
							.get(targetWord)[contentWordId] = embeddingMatrix
									.get(targetWord)[contentWordId] + 1;
				}
			}
		}

		return embeddingMatrix;
	}

    /**
     * Modification from the above model where
     * the distance between two words in a sentence is also considered.
     *
     * @param data_path
     * @param target_vocab
     * @param embedding_map
     * @return
     */
    private static HashMap<String, float[]> getEmbeddingsD(String dataPath,
                                                          HashMap<String, Integer> contentVocab, Set<String> targetVocab, boolean firstScore) {
        
        final HashMap<String, float[]> embeddingMatrix = new HashMap<String, float[]>();
        for (final String target_word : targetVocab) {
            embeddingMatrix.put(target_word, new float[contentVocab.size()]);
        }
        
        final Collection<List<String>> sentenceCollection = SentenceCollection.Reader
        .readSentenceCollection(dataPath);
        
        for (final List<String> sentence : sentenceCollection) {
            final int len = sentence.size();
            for (int i = 0; i < len; i++) {
                final String wordi = sentence.get(i);
                if (targetVocab.contains(wordi)) {
                    for (int j = 0; j < len; j++) {
                        final String wordj = sentence.get(j);
                        if (contentVocab.containsKey(wordj)) {
                            final int contentWordId = contentVocab.get(wordj);
                            final float score;
                            if (firstScore) {
                                score = (float) ((len - Math.abs(i - j) + 0.0) / len);
                            } else {
                                score = (float) (1.0 / (Math.abs(i - j) + 1.0));
                            }
                            embeddingMatrix.get(wordi)[contentWordId]
                            = embeddingMatrix.get(wordi)[contentWordId] + score;
                        }
                    }
                }
            }
        }
        
        return embeddingMatrix;
    }
    
    /**
     * considers dependency-based contexts
     *
     * @param data_path
     * @param target_vocab
     * @param embedding_map
     * @return
     */
    private static Pair<HashMap<String, float[]>, Integer> getDepEmbeddings(String dataPath,
                                                                            HashMap<String, Integer> contentVocab,
                                                                            Set<String> targetVocab) {
        
        final ArrayList<String> targetVocabList = new ArrayList<String>(targetVocab);
        final int targetVocabSize = targetVocabList.size();
        
        final ArrayList<ArrayList<Integer>> contexts = new ArrayList<ArrayList<Integer>>();
        for (int i = 0; i < targetVocabSize; i++) {
            contexts.add(new ArrayList<Integer>());
        }
        
        final HashId<String> contextIds = new HashId<String>();
        
        final Collection<List<Conll>> conllCollection = ConllCollection.Reader
        .readConllCollection(dataPath);
        
        for (final List<Conll> conlls : conllCollection) {
            final ArrayList<String> sentenceWords = new ArrayList<String>();
            if (contentVocab != null) {
                for (final Conll conlli : conlls) {
                    if (contentVocab.containsKey(conlli.word)) {
                        sentenceWords.add(conlli.word);
                    }
                }
            }
            for (final Conll conlli : conlls) {
                int wordIndex = targetVocabList.indexOf(conlli.word);
                if (wordIndex != -1) {
                    ArrayList<Integer> wordContexts = contexts.get(wordIndex);
                    
                    // add parent context
                    String parent;
                    if (conlli.head == 0) {
                        parent = "<ROOT>";
                    } else {
                        parent = conlls.get(conlli.head - 1).word;
                    }
                    //wordContexts.add(contextIds.get(parent));
                    //wordContexts.add(contextIds.get(conlli.rel + "-1"));
                    wordContexts.add(contextIds.get(parent + "/" + conlli.rel + "-1"));
                    
                    // add children contexts
                    final HashMap<Integer, String> grandchildren = new HashMap<Integer, String>();
                    for (final Conll conllj : conlls) {
                        if (conllj.head == conlli.index) {
                            if (conllj.rel == "prep") {
                                grandchildren.put(conllj.index, "prep_" + conllj.word);
                            }
                            else {
                                //wordContexts.add(contextIds.get(conllj.word));
                                //wordContexts.add(contextIds.get(conllj.rel));
                                wordContexts.add(contextIds.get(conllj.word + "/" + conllj.rel));
                            }
                        }
                    }
                    
                    // add grandchildren contexts
                    for (final Conll conllj : conlls) {
                        if (conllj.rel == "pobj") {
                            String new_rel = grandchildren.remove(conllj.head);
                            //wordContexts.add(contextIds.get(conllj.word));
                            //wordContexts.add(contextIds.get(new_rel));
                            wordContexts.add(contextIds.get(conllj.word + "/" + new_rel));
                        }
                    }
                    assert grandchildren.isEmpty();
                    
                    // add co-occurences
                    if (contentVocab != null) {
                        for (final String word : sentenceWords) {
                            wordContexts.add(contextIds.get(word));
                        }
                    }
                }
            }
        }
        
        final int contextSize = contextIds.size();
        
        final HashMap<String, float[]> embeddingMatrix = new HashMap<String, float[]>();
        for (final String target_word : targetVocab) {
            embeddingMatrix.put(target_word, new float[contextSize]);
        }
        
        for (int wordIndex = 0; wordIndex < targetVocabSize; wordIndex++) {
            float[] embedding = new float[contextSize];
            ArrayList<Integer> wordContexts = contexts.get(wordIndex);
            for (int contextId : wordContexts) {
                embedding[contextId] += 1.0;
            }
            embeddingMatrix.put(targetVocabList.get(wordIndex), embedding);
        }
        
        return new Pair<HashMap<String, float[]>, Integer>(embeddingMatrix, contextSize);
    }

    public static class ObjectiveFunction {
        HashMap<String, float[]> D;
        
        public double objective(double[] weights) {
            double sum = 0.0;
            
            for (final String word : D.keySet()) {
                float[] contexts = D.get(word);
                for (int c = 0; c < contexts.length; c++) {
                    if (contexts[c] > 0.0) {
                        sum += Math.log(1.0 / (1.0 + Math.exp(contexts[c] * weights[c])));
                    } else {
                        sum += Math.log(1.0 / (1.0 + Math.exp(-weights[c])));
                    }
                }
            }
            
            return sum;
        }
        
        public ObjectiveFunction(HashMap<String, float[]> D) {
            this.D = D;
        }
    }

	/**
	 * Read the core WordNet senses and map each to a unique integer. Used by
	 * the simple model below.
	 */
	private static HashMap<String, Integer> getWordNetVocab(
			String coreWordNetPath) throws Exception {
		final HashMap<String, Integer> vocab = new HashMap<String, Integer>();
		final BufferedReader reader = new BufferedReader(
				new FileReader(coreWordNetPath));
		String line = "";
		while ((line = reader.readLine()) != null) {
			final String[] parts = line.split(" ");
			final String word = parts[2].replace("[", "").replace("]", "");
			vocab.put(word, vocab.size());
		}
		reader.close();
		return vocab;
	}

	/**
	 * Get all of the words in the evaluation dataset.
	 *
	 * @param path
	 * @return
	 * @throws Exception
	 */
	private static Set<String> getWordSimVocab(String path) throws Exception {
		final Set<String> vocab = new HashSet<String>();
		final BufferedReader reader = new BufferedReader(new FileReader(path));
		String line = "";
		line = reader.readLine();
		final String[] keys = line.split(",");

		// Read the first line that contains the column keys.
		if (keys.length != 3) {
			System.out.println("There should be two words per line "
					+ "and a single score for each of these word "
					+ "pairs. We just saw, " + line);
			System.exit(0);
		}
		while ((line = reader.readLine()) != null) {
			final String[] parts = line.split(",");
			if (parts.length != 3) {
				System.out.println("WordSim line: " + line
						+ " should contain two words and a score.");
				System.exit(0);
			}
			final String word1 = parts[0];
			final String word2 = parts[1];
			vocab.add(word1);
			vocab.add(word2);
		}
		reader.close();
		return vocab;
	}

	/**
	 * Read the embedding parameters from a file.
	 *
	 * @param path
	 * @return
	 * @throws Exception
	 */
	private static HashMap<String, float[]> readEmbeddings(String path)
			throws Exception {
		final HashMap<String, float[]> embeddings = new HashMap<String, float[]>();
		final BufferedReader reader = new BufferedReader(new FileReader(path));
		String line = "";

		// Read the first line that contains the number of words and the
		// embedding dimension.
		line = reader.readLine().trim();

		String[] parts = line.split("\\s{1,}");
		if (parts.length < 2) {
			System.out.println("Format of embedding file wrong."
					+ "First line should contain number of words "
					+ "embedding dimension");
			System.exit(0);
		}
		final int vocab_size = Integer.parseInt(parts[0]);
		final int embedding_dim = Integer.parseInt(parts[1]);

		// Read the embeddings.
		int count_lines = 0;
		while ((line = reader.readLine()) != null) {
			if (count_lines > vocab_size) {
				System.out.println("Embedding file has more words than"
						+ "provided vocab size.");
				System.exit(0);
			}
			parts = line.split("\\s{1,}");
			final String word = parts[0];
			final float[] emb = new float[embedding_dim];
			for (int e_dim = 0; e_dim < embedding_dim; ++e_dim) {
				emb[e_dim] = Float.parseFloat(parts[e_dim + 1]);
			}
			embeddings.put(word, emb);
			++count_lines;
		}
		System.out.println("Read " + count_lines + " embeddings of dimension: "
				+ embedding_dim);
		reader.close();
		return embeddings;
	}

	/**
	 * Get a list of each of the word pair scores in the WordSim353 set. These
	 * pairs are not necessarily unique or symmetrical.
	 *
	 * @param path
	 * @return
	 * @throws Exception
	 */
	private static List<Pair<Pair<String, String>, Float>> readWordSimPairs(
			String path) throws Exception {
		final List<Pair<Pair<String, String>, Float>> wordSimPairs = new LinkedList<Pair<Pair<String, String>, Float>>();
		final BufferedReader reader = new BufferedReader(new FileReader(path));
		String line = "";
		line = reader.readLine();
		final String[] keys = line.split(",");

		// Read the first line that contains the column keys.
		if (keys.length != 3) {
			System.out.println("There should be two words per line "
					+ "and a single score for each of these word "
					+ "pairs. We just saw, " + line);
			System.exit(0);
		}
		while ((line = reader.readLine()) != null) {
			final String[] parts = line.split(",");
			if (parts.length != 3) {
				System.out.println("WordSim line: " + line
						+ " should contain two words and a score.");
				System.exit(0);
			}
			final String word1 = parts[0];
			final String word2 = parts[1];
			final Float score = Float.parseFloat(parts[2]);

			// Check that each pair is only included once, regardless of the
			// word order
			// in the example.
			final Pair<String, String> wordPair = new Pair<String, String>(
					word1, word2);
			wordSimPairs.add(
					new Pair<Pair<String, String>, Float>(wordPair, score));
		}
		reader.close();
		return wordSimPairs;
	}

	/*
	 * Reduce the embeddings vocabulary to only the words that will be needed
	 * for the word similarity task.
	 */
	private static HashMap<String, float[]> reduceVocabulary(
			HashMap<String, float[]> embeddings, Set<String> targetVocab) {
		final HashMap<String, float[]> prunedEmbeddings = new HashMap<String, float[]>();
		for (final String word : targetVocab) {
			if (embeddings.containsKey(word)) {
				prunedEmbeddings.put(word, embeddings.get(word));
			}
		}
		return prunedEmbeddings;
	}

	/**
	 * Calculate spearmans rho on the wordSim353 dataset (or any other dataset
	 * with similar formatting).
	 *
	 * @param wordSimPairs
	 * @param wordEmbeddings
	 * @return
	 * @throws Exception
	 */
	private static double spearmansScore(
			List<Pair<Pair<String, String>, Float>> wordSimPairs,
			HashMap<String, float[]> wordEmbeddings) throws Exception {

		final double[] predictions = new double[wordSimPairs.size()];
		final double[] labels = new double[wordSimPairs.size()];
		int pairNum = 0;
		for (final Pair<Pair<String, String>, Float> wordPair : wordSimPairs) {
			// Find the cosine of the word embeddings.
			final String word1 = wordPair.getFirst().getFirst();
			final String word2 = wordPair.getFirst().getSecond();
			if (wordEmbeddings.containsKey(word1)
					&& wordEmbeddings.containsKey(word2)) {
				predictions[pairNum] = cosineSimilarity(
						wordEmbeddings.get(word1), wordEmbeddings.get(word2));
			} else {
				// Unmodelled words have 0.5 similarity.
				predictions[pairNum] = 0.5;
			}
			labels[pairNum] = wordPair.getSecond();
			pairNum++;
		}

		final NaturalRanking ranking = new NaturalRanking(NaNStrategy.REMOVED);
		final SpearmansCorrelation spearman = new SpearmansCorrelation(ranking);

		return spearman.correlation(predictions, labels);
	}

	/**
	 * Write embeddings to a file.
	 *
	 * @param embeddings
	 * @param embeddingPath
	 * @param embeddingDim
	 * @throws Exception
	 */
	private static void writeEmbeddings(HashMap<String, float[]> embeddings,
			String path, int embeddingDim) throws Exception {
		final BufferedWriter writer = new BufferedWriter(new FileWriter(path));
		writer.write(embeddings.size() + " " + embeddingDim + "\n");
		for (final Map.Entry<String, float[]> wordEmbedding : embeddings
				.entrySet()) {
			final String word = wordEmbedding.getKey();
			final String embeddingString = Arrays
					.toString(wordEmbedding.getValue()).replace(", ", " ")
					.replace("[", "").replace("]", "");
			if (wordEmbedding.getValue().length != embeddingDim) {
				System.out.println("The embedding for " + word + " is not "
						+ embeddingDim + "D.");
				System.exit(0);
			}
			writer.write(word + " " + embeddingString + "\n");
		}
		writer.close();
	}
}