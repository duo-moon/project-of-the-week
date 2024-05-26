```mermaid
flowchart TD
	node1["data/salaries.csv.dvc"]
	node2["eval"]
	node3["process"]
	node4["train"]
	node1-->node3
	node3-->node2
	node3-->node4
	node4-->node2
```
