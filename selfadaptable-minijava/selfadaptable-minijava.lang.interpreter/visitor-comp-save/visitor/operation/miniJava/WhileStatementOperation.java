package visitor.operation.miniJava;

import miniJava.visitor.miniJava.State;

public interface WhileStatementOperation extends StatementOperation {
	void evaluateStatement(State state);
}
