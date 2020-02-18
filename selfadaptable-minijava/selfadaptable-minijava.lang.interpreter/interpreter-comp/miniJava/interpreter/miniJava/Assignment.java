package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface Assignment extends EObject, Statement {
	Assignee getAssignee();

	void setAssignee(Assignee value);

	Expression getValue();

	void setValue(Expression value);

	void evaluateStatement(State state);
}
