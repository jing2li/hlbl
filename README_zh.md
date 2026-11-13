Quick note: 

This branch exist to gauge the difference between using zerohalf function on KQED kernel vs using normal periodic wrapping

# Procedures #
1. allocate additional memory to store both zerohalf and non-zerohalf result 
2. geometry becomes 8: P234 is main's P2_0, P2_1, P3, P4_0, P4_1 (minimum image), followed by zerohalf P2_0, P2_1, P3 (geometry 5-7).
   P4 has no zerohalf copy: the earlier cvc runs did not compute it, so there is nothing to compare.
3. modify kernel sum functions to allow this