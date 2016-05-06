package edu.cornell.cs.nlp.assignments.util;

import java.io.Serializable;
import java.util.HashMap;

/**
 *
 * @author Brittany Nkounkou
 */
public class HashId<S extends Serializable> {
    private final HashMap<S, Integer> hashmap;
    
	public HashId() {
		this.hashmap = new HashMap<S, Integer>();
	}
    
    public int get(S s) {
        if (!hashmap.containsKey(s)) {
            hashmap.put(s, hashmap.size());
        }
        return hashmap.get(s);
    }
    
    public int size() {
        //System.out.println(hashmap);
        return hashmap.size();
    }
}