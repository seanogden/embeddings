package edu.cornell.cs.nlp.assignments;

/**
 * @author Brittany Nkounkou
 *
 */
public class Conll {
    public final int index;        // index in sentence
    public final String word;      // word itself
    public final String cpos;      // coarse-grained part-of-speech
    public final String fpos;      // fine-grained part-of-speech
    public final int head;         // index of parent in dependency tree
    public final String rel;       // relation to parent in dependency tree
    
    public Conll(int i, String w, String c, String f, int h, String r) {
        index = i;
        word = w;
        cpos = c;
        fpos = f;
        head = h;
        rel = r;
    }
}