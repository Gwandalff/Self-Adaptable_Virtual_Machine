package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface Assignment extends EObject, AcceptInterface, Statement {
	Assignee getAssignee();

	void setAssignee(Assignee value);

	Expression getValue();

	void setValue(Expression value);
}
