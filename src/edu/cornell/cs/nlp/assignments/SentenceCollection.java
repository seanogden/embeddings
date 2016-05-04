package edu.cornell.cs.nlp.assignments;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.AbstractCollection;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * @author Yoav Artzi
 *
 */
public class SentenceCollection extends AbstractCollection<List<String>> {
	String fileName;

	public SentenceCollection(String fileName) {
		this.fileName = fileName;
	}

	@Override
	public Iterator<List<String>> iterator() {
		try {
			final BufferedReader reader = new BufferedReader(
					new FileReader(fileName));
			return new SentenceIterator(reader);
		} catch (final FileNotFoundException e) {
			throw new RuntimeException(
					"Problem with SentenceIterator for " + fileName);
		}
	}

	@Override
	public int size() {
		int size = 0;
		final Iterator<List<String>> i = iterator();
		while (i.hasNext()) {
			size++;
			i.next();
		}
		return size;
	}

	public static class Reader {
		static Collection<List<String>> readSentenceCollection(
				String fileName) {
			return new SentenceCollection(fileName);
		}
	}

	static class SentenceIterator implements Iterator<List<String>> {

		BufferedReader reader;

		public SentenceIterator(BufferedReader reader) {
			this.reader = reader;
		}

		@Override
		public boolean hasNext() {
			try {
				return reader.ready();
			} catch (final IOException e) {
				return false;
			}
		}

		@Override
		public List<String> next() {
			try {
				final String line = reader.readLine();
				final String[] words = line.split("\\s+");
				final List<String> sentence = new ArrayList<String>();
				for (int i = 0; i < words.length; i++) {
					final String word = words[i];
					sentence.add(word.toLowerCase());
				}
				return sentence;
			} catch (final IOException e) {
				throw new NoSuchElementException();
			}
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}
	}

}