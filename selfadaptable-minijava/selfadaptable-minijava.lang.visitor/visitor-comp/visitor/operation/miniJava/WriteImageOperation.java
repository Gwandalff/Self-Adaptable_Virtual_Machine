package visitor.operation.miniJava;

import miniJava.visitor.miniJava.State;

public interface WriteImageOperation extends StatementOperation {
	void evaluateStatement(State state);
}
