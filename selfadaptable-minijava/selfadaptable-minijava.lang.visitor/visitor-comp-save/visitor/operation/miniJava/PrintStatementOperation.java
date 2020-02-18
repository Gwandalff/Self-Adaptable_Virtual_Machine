package visitor.operation.miniJava;

import miniJava.visitor.miniJava.State;

public interface PrintStatementOperation extends StatementOperation {
	void evaluateStatement(State state);
}
