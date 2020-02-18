package visitor.operation.miniJava;

import miniJava.visitor.miniJava.State;

public interface ReturnOperation extends StatementOperation {
	void evaluateStatement(State state);
}
