package visitor.operation.miniJava;

import miniJava.visitor.miniJava.State;

public interface BlockOperation extends StatementOperation {
	void evaluateStatementKeepContext(State state);

	void evaluateStatement(State state);
}
