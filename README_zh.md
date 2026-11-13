Quick note: 

This branch exist to gauge the difference between using zerohalf function on KQED kernel vs using normal periodic wrapping

# Procedures #
1. allocate additional memory to store both zerohalf and non-zerohalf result 
2. geometry becomes 6, p23 length doubles, with correct p23 followed by zerohalf p23
3. modify kernel sum functions to allow this