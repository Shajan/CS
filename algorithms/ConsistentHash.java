import java.util.*;
import java.util.regex.*;

class ConsistentHash {
  TreeMap<Integer, String> nodes;    // Multiple entries/machine
  int multiplier;                    // Number of nodes for each container
  String salt;                       // Add this salt before hashing the key
                                     // Note that consistent hashing helps with spreading load on cache machines
                                     // Salting the keys helps spread out backend load when cache machines fail

  Map<String, Integer> saltHash;     // Memoize salt hash to reduce recompute
  boolean debug=true;

  public static void main(String[] args) {
    ConsistentHash ch = new ConsistentHash(args, 7, "foobar");

    if (args.length > 1) {
      for (int i=0; i<5; ++i) {
         System.out.println(ch.getContainer("abc" + i));
         System.out.println(ch.getContainer("xxx" + i));
      }
    }
  }

  public ConsistentHash(String[] containers, int multiplier, String salt) {
    this.multiplier = multiplier;
    this.salt = salt;
    this.saltHash = new HashMap<String, Integer>();
    hashContainers(containers);
  }

  private void hashContainers(String[] containers) {
    this.nodes = new TreeMap<Integer, String>();
    for (String container : containers) {
      addContainer(container);
    }

    if (debug) {
      for (Map.Entry<Integer, String> entry : nodes.entrySet()) {
        System.out.println(entry);
      }
    }
  }

  public void addContainer(String container) {
    // Note: Java String.hashCode does not give good distribution
    // Improve this part with a better hash function
    for (int i=0; i<multiplier; ++i) {
      nodes.put(getHash(container, i), container);
    }
  }

  public void removeContainer(String container) {
    for (int i=0; i<multiplier; ++i) {
      nodes.remove(getHash(container, i));
    }
  }

  // Given a key, get the container
  public String getContainer(String key) {
    int hash = getHash(key, salt); 

    if (debug)
      System.out.println(String.format("%s -> %d", key, hash));

    // Find the node that is equal to the hash value (unlikely to have one), or one just following it.
    // If there are no hasvalues greater than hash, then pick the first hash value in the nodes map
    // The hasmap values form a circular namespace
    Map.Entry<Integer, String> entry = nodes.ceilingEntry(hash);

    if (debug)
      System.out.println("ceiling:" + entry);

    // Handle the case where 'hash' is greater than all keys in 'nodes'
    if (entry == null) {
      entry = nodes.firstEntry();
    }

    return entry.getValue();
  }

  // Hashing helper routines
  private int getHash(String s) {
    // java string.hashCode sucks wrt. distribution of hash values
    //int hash = s.hashCode();
    int hash = 7;
    for (int i=0; i <s.length(); i++) {
      hash = (hash*31) ^ s.charAt(i);
      hash ^= (hash >>> 20) ^ (hash >>> 12);
    }
    return hash ^ (hash >>> 7) ^ (hash >>> 4); 
  }

  private int getHash(String s, int salt) {
    return getHash(s, Integer.toString(salt));
  }

  private int getHash(String s, String salt) {
    int sHash = 0;
    if (!saltHash.containsKey(salt)) {
      sHash = getHash(salt);
      saltHash.put(salt, sHash);
    }
    return (s.hashCode() ^ sHash);
  }
}
