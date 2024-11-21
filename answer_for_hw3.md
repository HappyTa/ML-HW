# Answers for [HW 3](./questions/princML_hw03-1.pdf)

## Question 1

### Part A

***i*** : How can we keep track of the node's purpose in the code with this boolean/binary variable? Because decision nodes are always the leaves of the tree, how could we check the type of node without a specific variable?

- For my implementation, I used a binary variable called `node_type` to keep track of the node's type. 0 represent a query node, 1 represent a decision node.
- If we did not have `node_type` available, we can check if the node have any children node, if not then it is probably a decision node.

***ii***: Why is it sufficient to only have one variable store either value rather than two variables for feature and label? How does this help save memory?

- It is sufficient to only have one variable for both feature and label because both value are of the same type and only 1 is needed at one time, so when a node is a decision node, we do not care about the feature, and when we are a query node, we do not care about the label.
- This can also save memory by requesting less memory, instead of need 2 extra int size memory chunk when the object is created, we only need 1 extra int size memory chunk.

***iii***: What do the key and value for this hash map/dictionary represent with respect to the decision tree node's children? How does the hash map/dictionary data structure save runtime during decision tree prediction when we have the response from the input feature (passed in as an argument)?

- The key in this hash map/dictionary will be the position, either left or right, the value will be a node object, representing the children nodes.
- Using a hash map/dictionary is great for runtime because any operation using this data type can result in O(1), greatly improve performance compare to other data structure.

### Part B


**Train function**
ii: How does changing the control flow only when the choice of a hyper-parameter reduce the amount of code we write if the rest of the control flow is identical regardless of hyper-parameter choice?

- By changing the control flow based on the hyper-parameter help reduce code duplication and from that reduce the amount of code we write.

iii: How does a recursive function build a tree when it only return a pointer to one decision tree node?

- The node that is return is the root of the tree and as the function recursively run, it will add children node to this root and slowly build out a tree.

**Predict Function**

iii: If we had access to a GPU could we perform each prediction in parallel; why or why not?

- We can definately perform the prediction in parallel, this is because each prediction is independent from each other.
