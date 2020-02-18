package visitor.operation.miniJava;

import miniJava.visitor.miniJava.State;

public interface IfStatementOperation extends StatementOperation {
	void evaluateStatement(State state);
}
