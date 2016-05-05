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
 * @author Brittany Nkounkou
 *
 */
public class ConllCollection extends AbstractCollection<List<Conll>> {
	String fileName;

	public ConllCollection(String fileName) {
		this.fileName = fileName;
	}

	@Override
	public Iterator<List<Conll>> iterator() {
		try {
			final BufferedReader reader = new BufferedReader(
					new FileReader(fileName));
			return new ConllIterator(reader);
		} catch (final FileNotFoundException e) {
			throw new RuntimeException(
					"Problem with ConllIterator for " + fileName);
		}
	}

	@Override
	public int size() {
		int size = 0;
		final Iterator<List<Conll>> i = iterator();
		while (i.hasNext()) {
			size++;
			i.next();
		}
		return size;
	}

	public static class Reader {
		static Collection<List<Conll>> readConllCollection(
				String fileName) {
			return new ConllCollection(fileName);
		}
	}

	static class ConllIterator implements Iterator<List<Conll>> {

		BufferedReader reader;

		public ConllIterator(BufferedReader reader) {
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
		public List<Conll> next() {
			try {
				final List<Conll> conlls = new ArrayList<Conll>();
                String line = reader.readLine();
                while (!line.equals("")) {
                    final String[] parts = line.split("\t");
                    conlls.add(new Conll(Integer.parseInt(parts[0]),
                                         parts[1],
                                         parts[3],
                                         parts[4],
                                         Integer.parseInt(parts[6]),
                                         parts[7]));
                    
                    line = reader.readLine();
                }
				return conlls;
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