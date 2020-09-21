package visitor.operation.miniJava;

import miniJava.visitor.miniJava.State;

public interface ForStatementOperation extends StatementOperation {
	void evaluateStatement(State state);
}
