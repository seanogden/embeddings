package edu.cornell.cs.nlp.assignments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Brittany Nkounkou
 *
 */
public class Conll {
    static final ArrayList<String> POS = new ArrayList<String>(Arrays.asList(new String[] {"PRP$", "VBG", "VBD", "``", "VBN", "POS", "''", "VBP", "WDT", "JJ", "WP", "VBZ", "DT", "#", "RP", "$", "NN", "FW", ",", ".", "TO", "PRP", "RB", ":", "NNS", "NNP", "VB", "WRB", "CC", "LS", "PDT", "RBS", "RBR", "CD", "EX", "IN", "WP$", "MD", "NNPS", "JJS", "JJR", "SYM", "UH"}));
    
    static final ArrayList<String> REL = new ArrayList<String>(Arrays.asList(new String[] {"cc", "number", "ccomp", "possessive", "prt", "num", "nsubjpass", "csubj", "conj", "amod", "nn", "neg", "discourse", "mark", "auxpass", "infmod", "mwe", "advcl", "aux", "ROOT", "prep", "parataxis", "nsubj", "rcmod", "advmod", "punct", "quantmod", "tmod", "acomp", "pcomp", "csubjpass", "poss", "npadvmod", "xcomp", "cop", "partmod", "dep", "appos", "det", "dobj", "pobj", "iobj", "expl", "predet", "preconj"}));
    
    public static final int POS_SIZE = POS.size();
    public static final int REL_SIZE = REL.size();
    
    public static int getPOSid(String pos) {
        int i = POS.indexOf(pos);
        if (i < 0) {
            throw new RuntimeException("Unknown POS " + pos);
        }
        return i;
    }
    
    public static int getRELid(String rel) {
        int i = REL.indexOf(rel);
        if (i < 0) {
            throw new RuntimeException("Unknown REL " + rel);
        }
        return i;
    }
    
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