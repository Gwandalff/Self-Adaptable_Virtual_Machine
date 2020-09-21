package visitor.operation.miniJava;

import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Value;

public interface ExpressionOperation extends StatementOperation, AssigneeOperation {
	void evaluateStatement(State state);

	Value evaluateExpression(State state);
}
