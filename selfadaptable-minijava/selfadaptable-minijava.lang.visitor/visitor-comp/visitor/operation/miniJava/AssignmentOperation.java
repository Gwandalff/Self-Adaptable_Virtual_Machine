package visitor.operation.miniJava;

import miniJava.visitor.miniJava.State;

public interface AssignmentOperation extends StatementOperation {
	void evaluateStatement(State state);
}
